from __future__ import annotations

"""exp_022：纯树基线实验（检验无泄漏下能否逼近 0.12）。

5 个变体，每个产出一份提交：
  1 lgbm_native_cat   LightGBM LambdaRank + 9 类别原生 categorical
  2 catboost_yetirank CatBoost YetiRank + 9 类别 cat_features
  3 xgboost_pairwise   XGBoost rank:pairwise + 9 类别 enable_categorical
  4 ensemble_rank      1+2+3 截面排名平均
  5 lgbm_target_enc    LightGBM + 9 类别扩展窗口 target encoding

两阶段（PLAN §6/§7）：
  Phase A  train[486,2918) 训练 + valid[2918,3161) 早停 → best_iteration
  Phase B  train+valid[486,3161) 用 best_iteration×1.1 固定轮数重训（不再早停，valid 已入训练集）
  → test 分块推理 → 组内 rank → 提交网格

防泄漏红线见 PLAN.md §6。默认 cap=1024（与 exp016 同口径、内存安全）。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")

import lightgbm as lgb

from exp_016_unified_expert_fusion.src.ranking import group_rank, rank_ic
from exp_016_unified_expert_fusion.src.prediction_contract import vector_to_grid, validate_prediction

ROOT = Path(r"D:\google_dl\book\youanbei")
COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
TREE = ROOT / "03_cache" / "processed_data_v1" / "tree"
RESULT = ROOT / "04_results" / "exp_022_tree_full_baseline"

TRAIN_START, TRAIN_STOP = 486, 2918
VALID_START, VALID_STOP = 2918, 3161
TEST_START, TEST_STOP = 3161, 3603
TEST_TIME_POINTS = 442
STOCK_COUNT = 5282
NUM_COLS = 408
CAT_COLS = list(range(408, 417))  # 9 categories (cat_0..cat_8)
LABEL_GAIN = tuple(range(64))


def _mm(path):
    return np.load(path, mmap_mode="r")


def capped_indices(groups, cap):
    groups = np.asarray(groups, dtype=np.int64)
    if cap and cap > 0:
        idx, offset = [], 0
        for size in groups:
            size = int(size)
            take = min(size, cap)
            idx.append(offset + np.linspace(0, size - 1, take, dtype=np.int64))
            offset += size
        return np.concatenate(idx), np.minimum(groups, cap)
    return np.arange(int(groups.sum()), dtype=np.int64), groups.astype(np.int64)


def per_group_ic(pred, target, groups):
    scores, offset = [], 0
    for size in groups:
        size = int(size)
        scores.append(rank_ic(pred[offset:offset + size], target[offset:offset + size]))
        offset += size
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def prep_native_cat(X):
    """数值 + 9 类别(int32)，用于 LightGBM categorical_feature。mmap 只读时先复制。"""
    X = np.asarray(X, dtype=np.float32)
    if not X.flags.writeable:
        X = X.copy()
    X[:, CAT_COLS] = np.rint(X[:, CAT_COLS]).astype(np.float32)
    return X


def expanding_target_enc(train_X, train_y, train_groups, apply_X, cat_cols, prior=0.0, smoothing=20.0):
    """扩展窗口 target encoding，严格只用早于当前组的样本。"""
    train_X = np.asarray(train_X, dtype=np.float32)
    apply_X = np.asarray(apply_X, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.float64)
    groups = np.asarray(train_groups, dtype=np.int64)
    cats = np.rint(train_X[:, cat_cols]).astype(np.int64)
    apply_cats = np.rint(apply_X[:, cat_cols]).astype(np.int64)

    # 每列独立累积
    enc_train = np.zeros((cats.shape[0], len(cat_cols)), dtype=np.float64)
    sums = [{} for _ in cat_cols]
    counts = [{} for _ in cat_cols]
    offset = 0
    for size in groups:
        size = int(size)
        for k, col in enumerate(cat_cols):
            c = cats[offset:offset + size, k]
            yy = y[offset:offset + size]
            # 当前组用累计历史（不含当前组）
            hist_mean = np.array([sums[k].get(int(v), prior) / max(counts[k].get(int(v), 0), 1.0) if counts[k].get(int(v), 0) > 0 else prior for v in c])
            enc_train[offset:offset + size, k] = hist_mean
            # 累加当前组（供后续组使用）
            for v, val in zip(c, yy):
                sums[k][int(v)] = sums[k].get(int(v), 0.0) + val
                counts[k][int(v)] = counts[k].get(int(v), 0) + 1
        offset += size

    # 应用侧：用最终训练累计统计（不含 apply 自身）
    enc_apply = np.zeros((apply_cats.shape[0], len(cat_cols)), dtype=np.float64)
    for k in range(len(cat_cols)):
        for i, v in enumerate(apply_cats[:, k]):
            n = counts[k].get(int(v), 0)
            enc_apply[i, k] = (sums[k].get(int(v), 0.0) / max(n, 1.0)) if n > 0 else prior

    Xt = np.asarray(train_X, dtype=np.float32).copy()
    Xa = apply_X.copy()
    Xt[:, cat_cols] = enc_train.astype(np.float32)
    Xa[:, cat_cols] = enc_apply.astype(np.float32)
    return Xt, Xa


def lgbm_params(extra=None):
    p = {"objective": "lambdarank", "metric": "None", "verbosity": -1,
         "label_gain": list(LABEL_GAIN), "lambdarank_truncation_level": 1024,
         "learning_rate": 0.05, "num_leaves": 63, "seed": 42}
    if extra:
        p.update(extra)
    return p


def train_lgbm(X, rel, groups, Xv=None, relv=None, groupsv=None, categorical_feature=None, num_rounds=None):
    """Phase A：Xv 早停返回 best_iteration；Phase B：num_rounds 固定轮数（PLAN §6 ×1.1，不早停）。"""
    ds = lgb.Dataset(X, label=rel, group=groups, categorical_feature=categorical_feature, free_raw_data=True)
    if num_rounds is not None:
        model = lgb.train(lgbm_params(), ds, num_boost_round=num_rounds)
        return model, num_rounds

    def feval(preds, dataset):
        ic = per_group_ic(np.asarray(preds, np.float32), dataset.get_label(), dataset.get_group())
        return "rank_ic", ic, True

    dv = lgb.Dataset(Xv, label=relv, group=groupsv, categorical_feature=categorical_feature, reference=ds, free_raw_data=True)
    model = lgb.train(lgbm_params(), ds, num_boost_round=1000, valid_sets=[dv], feval=feval,
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    return model, model.best_iteration


def catboost_pool(X, rel, groups, cat_cols):
    """CatBoost 类别列必须是 int/str；用 FeaturesData 分离数值与类别列（与 exp016 一致）。"""
    import catboost as cb
    X = np.asarray(X, np.float32)
    numeric_idx = [i for i in range(X.shape[1]) if i not in cat_cols]
    numeric = np.ascontiguousarray(X[:, numeric_idx], dtype=np.float32)
    categorical = np.rint(X[:, cat_cols]).astype(np.int64).astype(str).astype(object)
    data = cb.FeaturesData(num_feature_data=numeric, cat_feature_data=categorical)
    group_id = None if groups is None else np.repeat(np.arange(len(groups), dtype=np.int64), groups)
    return cb.Pool(data, label=rel, group_id=group_id)


def train_catboost(X, rel, groups, Xv=None, relv=None, groupsv=None, cat_features=CAT_COLS, num_rounds=None):
    """Phase A：eval_set 早停；Phase B：num_rounds 固定轮数（不早停）。"""
    import catboost as cb
    params = {"loss_function": "YetiRank", "learning_rate": 0.05, "depth": 6,
              "verbose": False, "allow_writing_files": False, "random_seed": 42,
              "eval_metric": "NDCG"}
    p = catboost_pool(X, rel, groups, cat_features)
    if num_rounds is not None:
        params["iterations"] = num_rounds
        model = cb.train(p, params)
        return model, num_rounds

    params["iterations"] = 1000
    pv = catboost_pool(Xv, relv, groupsv, cat_features)
    model = cb.train(p, params, eval_set=pv, early_stopping_rounds=50)
    bi = model.get_best_iteration()
    if bi is None or int(bi) < 0:
        bi = 999
    return model, int(bi)


def train_xgboost(X, rel, groups, Xv=None, relv=None, groupsv=None, feature_types=None, num_rounds=None):
    """Phase A：evals 早停；Phase B：num_rounds 固定轮数（不早停）。"""
    import xgboost as xgb
    params = {"objective": "rank:pairwise", "tree_method": "hist", "verbosity": 0,
              "learning_rate": 0.05, "max_depth": 6, "seed": 42, "eval_metric": "ndcg",
              "ndcg_exp_gain": False}
    dm = xgb.DMatrix(X, label=rel, group=groups, enable_categorical=True, feature_types=feature_types)
    if num_rounds is not None:
        model = xgb.train(params, dm, num_boost_round=num_rounds)
        return model, num_rounds

    dv = xgb.DMatrix(Xv, label=relv, group=groupsv, enable_categorical=True, feature_types=feature_types)
    model = xgb.train(params, dm, num_boost_round=1000, evals=[(dv, "valid")],
                      early_stopping_rounds=50, verbose_eval=False)
    return model, model.best_iteration


def predict_chunked(model, X, kind, groups, chunk=200_000, prep_cats=False, feature_types=None):
    """分块推理。prep_cats=True 用于原始 mmap 输入（类别列 rint）；已编码/target-enc 数据禁用。"""
    parts = []
    for i in range(0, X.shape[0], chunk):
        Xb = np.asarray(X[i:i + chunk], dtype=np.float32)
        if prep_cats:
            if not Xb.flags.writeable:
                Xb = Xb.copy()
            Xb[:, CAT_COLS] = np.rint(Xb[:, CAT_COLS]).astype(np.float32)
        if kind == "lgbm":
            ni = getattr(model, "best_iteration", None)
            if not isinstance(ni, (int, np.integer)) or ni <= 0:
                ni = None
            parts.append(model.predict(Xb, num_iteration=ni))
        elif kind == "catboost":
            parts.append(model.predict(catboost_pool(Xb, None, None, CAT_COLS)))
        elif kind == "xgb":
            import xgboost as xgb
            parts.append(model.predict(xgb.DMatrix(Xb, enable_categorical=True, feature_types=feature_types)))
    return np.concatenate(parts).astype(np.float32)


def feature_types_str():
    ft = ['q'] * 419
    for i in CAT_COLS:
        ft[i] = 'c'
    return ft


def build_submission(pred_vec, test_time, test_stock, test_groups):
    ranked = group_rank(pred_vec, test_groups)
    grid = vector_to_grid(ranked, test_time, test_stock, TEST_START, TEST_TIME_POINTS)
    mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
    mask[np.asarray(test_time, dtype=np.int32) - TEST_START, np.asarray(test_stock, dtype=np.int32)] = True
    report = validate_prediction(grid, mask)
    return grid, report


def smoke():
    rng = np.random.default_rng(0)
    n, groups = 2400, np.full(24, 100, dtype=np.int32)
    X = rng.normal(size=(n, 419)).astype(np.float32)
    X[:, CAT_COLS] = rng.integers(0, 10, size=(n, 9)).astype(np.float32)
    rel = rng.integers(0, 64, size=n).astype(np.int32)
    Xro = X.copy()
    Xro.flags.writeable = False  # 模拟只读 mmap

    ft = feature_types_str()

    m, bi = train_lgbm(X, rel, groups, X, rel, groups, categorical_feature=CAT_COLS)
    p = predict_chunked(m, Xro, "lgbm", groups, prep_cats=True)
    assert p.shape == (n,) and np.isfinite(p).all()
    print("SMOKE_LGBM_A_OK best_iter=", bi)

    mB, nb = train_lgbm(X, rel, groups, categorical_feature=CAT_COLS, num_rounds=max(1, int(round(bi * 1.1))))
    p = predict_chunked(mB, Xro, "lgbm", groups, prep_cats=True)
    assert p.shape == (n,) and np.isfinite(p).all()
    print("SMOKE_LGBM_B_OK rounds=", nb)

    mc, bic = train_catboost(X, rel, groups, X, rel, groups, CAT_COLS)
    p = predict_chunked(mc, Xro, "catboost", groups)
    assert p.shape == (n,) and np.isfinite(p).all()
    print("SMOKE_CATBOOST_OK best_iter=", bic)

    mx, bix = train_xgboost(X, rel, groups, X, rel, groups, ft)
    p = predict_chunked(mx, Xro, "xgb", groups, prep_cats=True, feature_types=ft)
    assert p.shape == (n,) and np.isfinite(p).all()
    print("SMOKE_XGB_OK best_iter=", bix)

    Xte, Xap = expanding_target_enc(X, np.asarray(rel, np.float64), groups, Xro, CAT_COLS)
    assert Xte.shape == X.shape and np.isfinite(Xte).all() and np.isfinite(Xap).all()
    print("SMOKE_TARGET_ENC_OK")
    return True


def run(variants, cap):
    t0 = time.time()
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "models").mkdir(parents=True, exist_ok=True)

    trX, trY, trRel, trG = _mm(TREE / "train_X.npy"), _mm(COMMON / "train_y.npy"), _mm(COMMON / "train_relevance.npy"), _mm(COMMON / "train_group_sizes.npy")
    vaX, vaY, vaRel, vaG = _mm(TREE / "valid_X.npy"), _mm(COMMON / "valid_y.npy"), _mm(COMMON / "valid_relevance.npy"), _mm(COMMON / "valid_group_sizes.npy")
    teX = _mm(TREE / "test_X.npy")
    teTime = _mm(COMMON / "test_time.npy"); teStock = _mm(COMMON / "test_stock.npy"); teG = _mm(COMMON / "test_group_sizes.npy")

    # 训练侧（Phase A: [486,2918)）与最终侧（Phase B: [486,3161)）
    idx_a, grp_a = capped_indices(trG, cap)
    Xa = prep_native_cat(trX[idx_a]); ya = np.asarray(trY[idx_a], np.float32); ra = np.asarray(trRel[idx_a], np.int32)
    va_full = prep_native_cat(vaX); vY = np.asarray(vaY, np.float32); vR = np.asarray(vaRel, np.int32)

    # Phase B 训练数据：train(cap) + valid(cap)
    idx_v, grp_v = capped_indices(vaG, cap)
    va_cap = prep_native_cat(vaX[idx_v])
    Xb = np.concatenate([Xa, va_cap], axis=0)
    yb = np.concatenate([ya, np.asarray(vaY[idx_v], np.float32)], axis=0)
    rb = np.concatenate([ra, np.asarray(vaRel[idx_v], np.int32)], axis=0)
    gb = np.concatenate([grp_a, grp_v]).astype(np.int32)

    ft = feature_types_str()
    rows = []
    preds = {}   # variant -> test prediction vector
    valids = {}  # variant -> valid prediction vector

    for vi in variants:
        name = {1: "lgbm_native_cat", 2: "catboost_yetirank", 3: "xgboost_pairwise",
                4: "ensemble_rank", 5: "lgbm_target_enc"}[vi]
        tt = time.time()
        print(f"[exp022] variant {vi} {name} start", flush=True)
        if vi == 1:
            m, bi = train_lgbm(Xa, ra, grp_a, va_full, vR, vaG, categorical_feature=CAT_COLS)
            pred_a = predict_chunked(m, va_full, "lgbm", vaG)
            ic = per_group_ic(pred_a, vY, vaG)
            mB, _ = train_lgbm(Xb, rb, gb, categorical_feature=CAT_COLS,
                               num_rounds=max(1, int(round(bi * 1.1))))
            pred_t = predict_chunked(mB, teX, "lgbm", teG, prep_cats=True)
        elif vi == 2:
            m, bi = train_catboost(Xa, ra, grp_a, va_full, vR, vaG, CAT_COLS)
            pred_a = predict_chunked(m, va_full, "catboost", vaG)
            ic = per_group_ic(pred_a, vY, vaG)
            mB, _ = train_catboost(Xb, rb, gb, num_rounds=max(1, int(round(bi * 1.1))))
            pred_t = predict_chunked(mB, teX, "catboost", teG)
        elif vi == 3:
            m, bi = train_xgboost(Xa, ra, grp_a, va_full, vR, vaG, ft)
            pred_a = predict_chunked(m, va_full, "xgb", vaG, feature_types=ft)
            ic = per_group_ic(pred_a, vY, vaG)
            mB, _ = train_xgboost(Xb, rb, gb, feature_types=ft,
                                  num_rounds=max(1, int(round(bi * 1.1))))
            pred_t = predict_chunked(mB, teX, "xgb", teG, prep_cats=True, feature_types=ft)
        elif vi == 4:
            # 复用 1/2/3 的预测做截面排名平均
            if not all(k in preds for k in (1, 2, 3)):
                print("[exp022] ensemble 需要先跑 1,2,3，跳过", flush=True)
                continue
            pred_t = group_rank(preds[1], teG) + group_rank(preds[2], teG) + group_rank(preds[3], teG)
            pred_a = group_rank(valids[1], vaG) + group_rank(valids[2], vaG) + group_rank(valids[3], vaG)
            ic = per_group_ic(pred_a, vY, vaG)
            bi = 0
        elif vi == 5:
            # 直接复用已物化的 Xa/Xb（类别列已 rint，值与原 mmap 一致），避免重复大拷贝
            Xa_te, va_te = expanding_target_enc(Xa, ya, grp_a, va_full, CAT_COLS)
            m, bi = train_lgbm(Xa_te, ra, grp_a, va_te, vR, vaG, categorical_feature=[])
            pred_a = predict_chunked(m, va_te, "lgbm", vaG)
            ic = per_group_ic(pred_a, vY, vaG)
            del Xa_te, va_te, m
            Xb_te, te_te = expanding_target_enc(Xb, yb, gb, teX, CAT_COLS)
            mB, _ = train_lgbm(Xb_te, rb, gb, categorical_feature=[],
                               num_rounds=max(1, int(round(bi * 1.1))))
            pred_t = predict_chunked(mB, te_te, "lgbm", teG)
            del Xb_te, te_te, mB
        else:
            continue

        preds[vi] = pred_t
        valids[vi] = pred_a
        grid, contract = build_submission(pred_t, teTime, teStock, teG)
        np.save(RESULT / f"prediction_{vi}.npy", grid)
        rows.append({"variant": name, "prediction": f"prediction_{vi}.npy",
                     "valid_mean_rank_ic": ic, "best_iteration": bi,
                     "contract_ok": contract["finite"] and contract["non_evaluation_all_0_5"],
                     "elapsed_s": round(time.time() - tt, 1)})
        print(f"[exp022] {name}: valid IC={ic:.6f}, best_iter={bi}, {time.time()-tt:.1f}s", flush=True)

    with open(RESULT / "fold_results.csv", "w", encoding="utf-8") as f:
        f.write("variant,prediction,valid_mean_rank_ic,best_iteration,contract_ok,elapsed_s\n")
        for r in rows:
            f.write(f"{r['variant']},{r['prediction']},{r['valid_mean_rank_ic']:.10f},{r['best_iteration']},{r['contract_ok']},{r['elapsed_s']}\n")
    metrics = {"experiment": "exp_022_tree_full_baseline", "cap": cap, "variants": rows,
               "baseline_exp021_online": 0.116568, "target": 0.12, "elapsed_s": round(time.time() - t0, 1)}
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "experiment": "exp_022_tree_full_baseline", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_submission_overwritten": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp022] DONE in {time.time()-t0:.1f}s", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cap", type=int, default=1024, help="每截面股票抽样上限；0=全量")
    ap.add_argument("--variants", type=str, default="1,2,3,4,5")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    variants = [int(x) for x in args.variants.split(",") if x.strip()]
    run(variants, args.cap)


if __name__ == "__main__":
    main()
