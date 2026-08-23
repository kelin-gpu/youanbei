from __future__ import annotations

"""T1.2：cat_5 四种处理消融（exp_018）。

对高基数 cat_5（树视图第 413 列，train 内 4093 个取值）比较四种处理：
  1. native        —— 原始类别作为 LightGBM categorical 特征
  2. frequency     —— 训练侧频率编码（unseen 填 0）
  3. unknown_bucket—— 未见取值显式映射到单一 UNK 桶
  4. remove        —— 完全移除 cat_5（baseline）

固定 legacy_328 数值主线（tree 前 328 列），各训 16 轮 LightGBM LambdaRank，
Train-only，在统一全量 valid 上比较 mean RankIC，并产出 4 份提交文件。

训练用 stock_cap=1024 控制资源（与 exp016 一致），类别词表/频率在全量 train 上统计。
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import lightgbm as lgb

ROOT = Path(r"D:\google_dl\book\youanbei")
COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
TREE = ROOT / "03_cache" / "processed_data_v1" / "tree"
RESULT = ROOT / "04_results" / "exp_018_cat5_ablation"

CAP = 1024
LEGACY_N = 328
CAT5_COL = 413
LABEL_GAIN = tuple(range(64))
TEST_START = 3161
TEST_TIME_POINTS = 442
STOCK_COUNT = 5282


def lambdarank_params(**overrides):
    params = {"objective": "lambdarank", "metric": "None", "verbosity": -1,
              "label_gain": list(LABEL_GAIN), "lambdarank_truncation_level": 1024}
    params.update(overrides)
    return params


def group_rank(values, groups):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.empty_like(values)
    offset = 0
    for size in groups:
        size = int(size)
        result[offset:offset + size] = rankdata(values[offset:offset + size], method="average").astype(np.float32) / float(size)
        offset += size
    return result


def rank_ic(prediction, target):
    left = np.asarray(prediction, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    left, right = left[finite], right[finite]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def capped_indices(groups, cap):
    idx, offset = [], 0
    for size in groups:
        size = int(size)
        take = min(size, cap)
        idx.append(offset + np.linspace(0, size - 1, take, dtype=np.int64))
        offset += size
    return np.concatenate(idx)


def per_group_ic(pred, target, groups):
    scores, offset = [], 0
    for size in groups:
        size = int(size)
        scores.append(rank_ic(pred[offset:offset + size], target[offset:offset + size]))
        offset += size
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite))


def build_features(X, treatment, freq_map, vocab, unk_token):
    """返回 (X_feat float32, categorical_feature list)。X 为 rows×419 矩阵（内存数组或 mmap）。"""
    num = np.ascontiguousarray(X[:, :LEGACY_N], dtype=np.float32)
    if treatment == "remove":
        return num, []
    cat5 = np.rint(X[:, CAT5_COL]).astype(np.int64)
    if treatment == "native":
        col = cat5.astype(np.float32)
        cat_feature = [LEGACY_N]
    elif treatment == "frequency":
        col = np.asarray([freq_map.get(int(v), 0.0) for v in cat5], dtype=np.float32)
        cat_feature = []
    elif treatment == "unknown_bucket":
        mapped = np.where(np.isin(cat5, list(vocab)), cat5, unk_token).astype(np.float32)
        col = mapped
        cat_feature = [LEGACY_N]
    else:
        raise ValueError(treatment)
    out = np.empty((num.shape[0], LEGACY_N + 1), dtype=np.float32)
    out[:, :LEGACY_N] = num
    out[:, LEGACY_N] = col
    return out, cat_feature


def main():
    t0 = time.time()
    RESULT.mkdir(parents=True, exist_ok=True)

    train_X = np.load(TREE / "train_X.npy", mmap_mode="r")
    valid_X = np.load(TREE / "valid_X.npy", mmap_mode="r")
    test_X = np.load(TREE / "test_X.npy", mmap_mode="r")
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    train_rel = np.load(COMMON / "train_relevance.npy", mmap_mode="r")
    train_groups = np.load(COMMON / "train_group_sizes.npy", mmap_mode="r")
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    valid_groups = np.load(COMMON / "valid_group_sizes.npy", mmap_mode="r")
    test_time = np.load(COMMON / "test_time.npy", mmap_mode="r")
    test_stock = np.load(COMMON / "test_stock.npy", mmap_mode="r")
    test_groups = np.load(COMMON / "test_group_sizes.npy", mmap_mode="r")

    # 全量 train 上的 cat_5 词表与频率
    cat5_train = np.rint(train_X[:, CAT5_COL]).astype(np.int64)
    uniq, counts = np.unique(cat5_train, return_counts=True)
    vocab = set(uniq.tolist())
    freq_map = {int(u): float(c) / float(cat5_train.size) for u, c in zip(uniq, counts)}
    unk_token = int(uniq.max()) + 1

    # capped train
    idx = capped_indices(train_groups, CAP)
    train_rows = train_X[idx]
    train_y_c = np.asarray(train_y[idx], dtype=np.float32)
    train_rel_c = np.asarray(train_rel[idx], dtype=np.int32)
    train_groups_c = np.minimum(np.asarray(train_groups, dtype=np.int32), CAP)
    valid_groups_i = np.asarray(valid_groups, dtype=np.int32)
    test_groups_i = np.asarray(test_groups, dtype=np.int32)

    treatments = ["native", "frequency", "unknown_bucket", "remove"]
    results = []

    for ti, treatment in enumerate(treatments, start=1):
        tt = time.time()
        X_tr, cat_feat = build_features(train_rows, treatment, freq_map, vocab, unk_token)
        X_va, _ = build_features(valid_X, treatment, freq_map, vocab, unk_token)
        X_te, _ = build_features(test_X, treatment, freq_map, vocab, unk_token)

        ds = lgb.Dataset(X_tr, label=train_rel_c, group=train_groups_c, categorical_feature=cat_feat, free_raw_data=True)
        model = lgb.train(lambdarank_params(learning_rate=0.05, num_leaves=31, seed=42), ds, num_boost_round=16)

        va_pred = model.predict(X_va)
        va_ic = per_group_ic(va_pred, valid_y, valid_groups_i)

        te_pred = model.predict(X_te)
        ranked_te = group_rank(te_pred, test_groups_i)
        grid = np.full((TEST_TIME_POINTS, STOCK_COUNT), 0.5, dtype=np.float32)
        grid[np.asarray(test_time, dtype=np.int32) - TEST_START, np.asarray(test_stock, dtype=np.int32)] = ranked_te
        mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
        mask[np.asarray(test_time, dtype=np.int32) - TEST_START, np.asarray(test_stock, dtype=np.int32)] = True
        contract_ok = (grid.shape == (TEST_TIME_POINTS, STOCK_COUNT) and grid.dtype == np.float32
                       and bool(np.isfinite(grid).all()) and bool(np.all(grid[~mask] == np.float32(0.5))) and int(mask.sum()) == 2042538)

        model_path = RESULT / f"model_{treatment}.txt"
        model_path.write_text(model.model_to_string(), encoding="utf-8")
        np.save(RESULT / f"prediction_{ti}.npy", grid)
        results.append({"treatment": treatment, "prediction": f"prediction_{ti}.npy",
                        "valid_mean_rank_ic": va_ic, "contract_ok": contract_ok,
                        "elapsed_s": round(time.time() - tt, 1)})
        print(f"[exp018] {treatment}: valid IC={va_ic:.6f}, contract_ok={contract_ok} ({time.time()-tt:.1f}s)", flush=True)

    baseline = next(r["valid_mean_rank_ic"] for r in results if r["treatment"] == "remove")
    for r in results:
        r["delta_vs_remove"] = round(r["valid_mean_rank_ic"] - baseline, 6)

    with open(RESULT / "fold_results.csv", "w", encoding="utf-8") as f:
        f.write("treatment,valid_mean_rank_ic,delta_vs_remove,contract_ok,elapsed_s\n")
        for r in results:
            f.write(f"{r['treatment']},{r['valid_mean_rank_ic']:.10f},{r['delta_vs_remove']},{r['contract_ok']},{r['elapsed_s']}\n")

    metrics = {"cat5_col": CAT5_COL, "train_unique_categories": int(uniq.size),
               "cap": CAP, "num_boost_round": 16,
               "results": results, "baseline": "remove"}
    metadata = {"experiment": "exp_018_cat5_ablation", "task": "T1.2 cat_5 ablation",
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "final_submission_overwritten": False, "elapsed_s": round(time.time() - t0, 1)}
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp018] DONE in {time.time()-t0:.1f}s", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
