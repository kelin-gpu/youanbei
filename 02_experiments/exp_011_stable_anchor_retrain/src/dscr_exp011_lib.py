"""exp_011 稳定锚点重训实验核心库。

数据口径与 exp_003/exp_007/exp_009 完全一致：
- 特征视图：processed_data_v1 的 legacy_328（tree 视图前 328 列，因果性已由缓存清单校验）。
- 标签：common/*_relevance.npy（连续 Y1 截断到 [0,1] 后映射为 64 档相关度）。
- 分组：每个时间点为 lambdaRank group。
- 抽样：每个时间点确定性 linspace 抽取至多 1200 只股票。
- 参数：exp_003 调优参数（learning_rate=0.0228695, num_leaves=79, ...）。
- 指标：逐时间截面 Spearman RankIC 的均值；处理 NaN/Inf/并列/空组。
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

# ---------------------------------------------------------------------------
# 常量与全局配置
# ---------------------------------------------------------------------------
T = 3603
S = 5282
TRAIN_START = 486
VALID_START = 2918
VALID_STOP = 3161
TEST_START = 3161
TEST_STOP = 3603
VALID_TIME_POINTS = VALID_STOP - VALID_START          # 243
TEST_TIME_POINTS = TEST_STOP - TEST_START             # 442

FEATURE_STOP = 328          # legacy_328 列数
TREE_COUNT = 419            # tree 视图总列数
TRAIN_STOCK_CAP = 1200
RECENT_LOOKBACK = 1702
BASE_ROUNDS = 8             # 全历史锚点轮数（与 exp_003 一致）
RECENT_ROUNDS = 16          # 近期专家轮数（exp_009 走步选定）
WEIGHT_CANDIDATES = (0.25, 0.30, 0.35)
SEEDS = (42, 2026, 3407)
ANCHOR_EXPECTED_VALID_IC = 0.092940153549703
ANCHOR_REPRO_TOLERANCE = 0.0003

NUM_THREADS = max(1, (os.cpu_count() or 8) - 2)

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "None",
    "learning_rate": 0.0228695,
    "num_leaves": 79,
    "min_data_in_leaf": 147,
    "feature_fraction": 0.80936,
    "bagging_fraction": 0.647764,
    "bagging_freq": 1,
    "lambda_l1": 2.35724,
    "lambda_l2": 0.238705,
    "max_bin": 127,
    "label_gain": list(range(64)),
    "lambdarank_truncation_level": 1024,
    "verbosity": -1,
    "num_threads": NUM_THREADS,
}

# ---------------------------------------------------------------------------
# legacy_328 内部特征块（用于消融；列区间与 manifest numeric_names 完全一致）
# 顺序：40 raw + 20 rank + 80 lag(1/5/20/60) + 滚动块(窗口优先：mean5/std5/change5/
# mean20/std20/change20/mean60/std60/change60) + 4 lag_available + 3 coverage + 1 age
# ---------------------------------------------------------------------------
FEATURE_BLOCKS = {
    "raw_40": [(0, 40)],
    "rank_20": [(40, 60)],
    "lag_short": [(60, 100)],        # lag_1 + lag_5
    "lag_long": [(100, 140)],        # lag_20 + lag_60
    "roll_mean": [(140, 160), (200, 220), (260, 280)],
    "roll_std": [(160, 180), (220, 240), (280, 300)],
    "roll_change": [(180, 200), (240, 260), (300, 320)],
    "lag_available": [(320, 324)],
    "state": [(324, 328)],           # history_coverage*3 + stock_age
}

# 消融配置：name -> (keep block names)
ABLATION_CONFIGS = {
    "full_328": [b for b in FEATURE_BLOCKS],
    "no_rank": [b for b in FEATURE_BLOCKS if b != "rank_20"],
    "no_short_lag": [b for b in FEATURE_BLOCKS if b != "lag_short"],
    "no_long_lag": [b for b in FEATURE_BLOCKS if b != "lag_long"],
    "no_roll_mean": [b for b in FEATURE_BLOCKS if b != "roll_mean"],
    "no_roll_std": [b for b in FEATURE_BLOCKS if b != "roll_std"],
    "no_roll_change": [b for b in FEATURE_BLOCKS if b != "roll_change"],
    "no_state": [b for b in FEATURE_BLOCKS if b != "state"],
}


def feature_cols_for_config(config_name: str) -> np.ndarray:
    blocks = ABLATION_CONFIGS[config_name]
    cols = [np.arange(start, stop) for block in blocks for start, stop in FEATURE_BLOCKS[block]]
    # 排序保证列序与 exp_003/exp_009 的 [0:328) 规范序一致（列序影响 feature_fraction 抽样结果）
    return np.sort(np.concatenate(cols)).astype(np.int64)


# ---------------------------------------------------------------------------
# 文件工具
# ---------------------------------------------------------------------------
def file_sha256(path, block_size=16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with open(partial, "wb") as handle:
        np.save(handle, array)
    os.replace(partial, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(partial, index=False, encoding="utf-8-sig")
    os.replace(partial, path)


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# RankIC 与评分
# ---------------------------------------------------------------------------
def rank_ic(prediction, target) -> float:
    """单个时间截面的 Spearman RankIC。"""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    usable = np.isfinite(prediction) & np.isfinite(target)
    if int(np.count_nonzero(usable)) < 2:
        return float("nan")
    x = rankdata(prediction[usable], method="average")
    y = rankdata(target[usable], method="average")
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def group_rank_ic_series(prediction, target, groups) -> np.ndarray:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    values = []
    offset = 0
    for size in groups:
        size = int(size)
        values.append(rank_ic(prediction[offset:offset + size], target[offset:offset + size]))
        offset += size
    assert offset == prediction.size == target.size, (offset, prediction.size, target.size)
    return np.asarray(values, dtype=np.float64)


def score_prediction(prediction, target, groups) -> dict:
    """按官方口径统计逐时间截面 RankIC 的全部指标。"""
    per_time = group_rank_ic_series(prediction, target, groups)
    finite = per_time[np.isfinite(per_time)]
    n = finite.size
    half = n // 2
    quarters = np.array_split(finite, 4)
    out = {
        "valid_time_count": int(n),
        "mean_rankic": float(np.nanmean(per_time)),
        "median_rankic": float(np.nanmedian(per_time)),
        "rankic_std": float(np.nanstd(per_time)),
        "icir": float(np.nanmean(per_time) / np.nanstd(per_time)) if np.nanstd(per_time) > 0 else float("nan"),
        "positive_ratio": float(np.nanmean(per_time > 0)),
        "first_half_rankic": float(np.nanmean(per_time[:half])),
        "second_half_rankic": float(np.nanmean(per_time[half:])),
        "worst_quarter_rankic": float(min(np.nanmean(q) for q in quarters)) if n else float("nan"),
    }
    for idx, q in enumerate(quarters):
        out[f"quarter_{idx + 1}_rankic"] = float(np.nanmean(q))
    return out


def group_rank_transform(prediction, groups) -> np.ndarray:
    """每个时间截面内转换为 (0,1] 百分位秩。"""
    prediction = np.asarray(prediction)
    ranked = np.empty(prediction.size, dtype=np.float32)
    offset = 0
    for size in groups:
        size = int(size)
        ranked[offset:offset + size] = (
            rankdata(prediction[offset:offset + size], method="average").astype(np.float32) / float(size)
        )
        offset += size
    assert offset == prediction.size
    return ranked


def rank_ic_self_test() -> dict:
    """手工样本验证 RankIC 实现。"""
    results = {}
    # 1. 完全正相关 -> 1.0
    x = np.arange(10, dtype=np.float64)
    results["identity"] = rank_ic(x, x)
    # 2. 完全负相关 -> -1.0
    results["reverse"] = rank_ic(x, x[::-1])
    # 3. 与 scipy.stats.spearmanr 对照
    rng = np.random.default_rng(123)
    a = rng.normal(size=50)
    b = a + 0.5 * rng.normal(size=50)
    from scipy.stats import spearmanr
    results["vs_spearmanr"] = rank_ic(a, b) - spearmanr(a, b).statistic
    # 4. 常量预测 -> nan
    results["constant_pred"] = rank_ic(np.ones(5), np.arange(5))
    # 5. 少于 2 只 -> nan
    results["too_few"] = rank_ic(np.array([1.0]), np.array([0.5]))
    # 6. NaN/Inf 过滤
    p = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.inf])
    t = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    results["nan_filter"] = rank_ic(p, t)
    # 7. 并列排名用 average 方法（Spearman 平局修正）
    p2 = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0])
    t2 = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.4])
    results["ties"] = rank_ic(p2, t2)
    assert abs(results["identity"] - 1.0) < 1e-12, results
    assert abs(results["reverse"] + 1.0) < 1e-12, results
    assert abs(results["vs_spearmanr"]) < 1e-12, results
    assert np.isnan(results["constant_pred"]) and np.isnan(results["too_few"])
    assert np.isfinite(results["nan_filter"]) and np.isfinite(results["ties"])
    return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)) for k, v in results.items()}


# ---------------------------------------------------------------------------
# 数据访问（processed_data_v1）
# ---------------------------------------------------------------------------
class Dataset:
    def __init__(self, dataset_dir, check_sha256: bool = True):
        self.dir = Path(dataset_dir)
        self.ready_path = self.dir / "READY"
        self.manifest_path = self.dir / "manifest.json"
        assert self.ready_path.exists(), "READY 缺失"
        assert self.manifest_path.exists(), "manifest.json 缺失"
        self.ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        assert self.manifest["status"] == "ready"
        manifest_sha = file_sha256(self.manifest_path)
        assert self.ready["manifest_sha256"] == manifest_sha, "manifest SHA-256 与 READY 不一致"
        if check_sha256:
            self._check_contract()
        self._load()

    def _check_contract(self):
        m = self.manifest
        assert m["dimensions"] == {"time": 3603, "stock": 5282, "raw_numeric": 99, "raw_category": 9}
        assert m["splits"]["train"]["start"] == TRAIN_START
        assert m["splits"]["train"]["stop"] == VALID_START
        assert m["splits"]["valid"]["start"] == VALID_START
        assert m["splits"]["valid"]["stop"] == VALID_STOP
        assert m["splits"]["test"]["start"] == TEST_START
        assert m["splits"]["test"]["stop"] == TEST_STOP
        assert m["features"]["legacy_numeric_prefix"] == FEATURE_STOP
        assert m["features"]["tree_count"] == TREE_COUNT
        assert m["legacy_compatibility"]["status"] == "passed"
        assert m["validation"]["test_mask"]["count"] == 2_042_538
        assert m["validation"]["test_mask"]["matches_official"] is True

    def _load(self):
        common_dir = self.dir / "common"
        tree_dir = self.dir / "tree"
        self.common = {}
        for split in ("train", "valid", "test"):
            entry = {
                "time": np.load(common_dir / f"{split}_time.npy", mmap_mode="r"),
                "stock": np.load(common_dir / f"{split}_stock.npy", mmap_mode="r"),
                "groups": np.load(common_dir / f"{split}_group_sizes.npy", mmap_mode="r"),
            }
            if split != "test":
                entry["y"] = np.load(common_dir / f"{split}_y.npy", mmap_mode="r")
                entry["relevance"] = np.load(common_dir / f"{split}_relevance.npy", mmap_mode="r")
            self.common[split] = entry
        self.tree = {
            split: np.load(tree_dir / f"{split}_X.npy", mmap_mode="r")
            for split in ("train", "valid", "test")
        }
        for split, matrix in self.tree.items():
            expected = int(self.manifest["expected_rows"][split])
            assert matrix.shape == (expected, TREE_COUNT), (split, matrix.shape)

    def split_meta(self) -> pd.DataFrame:
        rows = []
        for split in ("train", "valid", "test"):
            values = self.common[split]
            rows.append({
                "split": split,
                "rows": int(values["time"].size),
                "time_start": int(values["time"][0]),
                "time_stop": int(values["time"][-1]) + 1,
                "time_points": int(values["groups"].size),
                "group_sum": int(values["groups"].sum()),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 索引、训练、预测
# ---------------------------------------------------------------------------
def row_slice_for_times(ds: Dataset, split: str, start_time: int, stop_time: int):
    times = ds.common[split]["time"]
    start = int(np.searchsorted(times, start_time, side="left"))
    stop = int(np.searchsorted(times, stop_time, side="left"))
    split_start = int(times[0])
    groups = np.asarray(ds.common[split]["groups"][start_time - split_start:stop_time - split_start], dtype=np.int32)
    assert int(groups.sum()) == stop - start
    return slice(start, stop), groups


def capped_indices_for_split(ds: Dataset, split: str, start_time: int, stop_time: int, cap: int):
    rows, groups = row_slice_for_times(ds, split, start_time, stop_time)
    if cap <= 0:
        return np.arange(rows.start, rows.stop, dtype=np.int64), groups
    capped = np.minimum(groups, int(cap)).astype(np.int32)
    indices = np.empty(int(capped.sum()), dtype=np.int64)
    source_offset, target_offset = int(rows.start), 0
    for full_size, capped_size in zip(groups, capped):
        full_size = int(full_size)
        capped_size = int(capped_size)
        positions = np.linspace(0, full_size - 1, capped_size, dtype=np.int64)
        indices[target_offset:target_offset + capped_size] = source_offset + positions
        source_offset += full_size
        target_offset += capped_size
    assert source_offset == int(rows.stop)
    assert target_offset == indices.size == int(capped.sum())
    return indices, capped


def stratified_capped_indices(ds: Dataset, split: str, start_time: int, stop_time: int, cap: int, rng):
    """按 64 档相关度分层抽样（可选实验）。"""
    rows, groups = row_slice_for_times(ds, split, start_time, stop_time)
    capped = np.minimum(groups, int(cap)).astype(np.int32)
    indices = np.empty(int(capped.sum()), dtype=np.int64)
    rel = ds.common[split]["relevance"]
    source_offset, target_offset = int(rows.start), 0
    for full_size, capped_size in zip(groups, capped):
        full_size = int(full_size)
        capped_size = int(capped_size)
        local = np.arange(source_offset, source_offset + full_size, dtype=np.int64)
        if capped_size < full_size:
            labels = np.asarray(rel[local], dtype=np.int32)
            order = np.argsort(labels, kind="stable")
            sorted_local = local[order]
            # 均匀取位置（等价于按标签分层确定性抽样）
            positions = np.linspace(0, full_size - 1, capped_size, dtype=np.int64)
            indices[target_offset:target_offset + capped_size] = sorted_local[positions]
        else:
            indices[target_offset:target_offset + capped_size] = local
        source_offset += full_size
        target_offset += capped_size
    return indices, capped


def build_training_arrays(ds: Dataset, segments, cap: int, cols: np.ndarray | None = None):
    if cols is None:
        cols = np.arange(FEATURE_STOP, dtype=np.int64)
    prepared = []
    total_rows = 0
    for split, start_time, stop_time in segments:
        indices, groups = capped_indices_for_split(ds, split, start_time, stop_time, cap)
        prepared.append((split, indices, groups))
        total_rows += indices.size
    X = np.empty((total_rows, cols.size), dtype=np.float32)
    y = np.empty(total_rows, dtype=np.int8)
    all_groups = []
    offset = 0
    for split, indices, groups in prepared:
        stop = offset + indices.size
        X[offset:stop] = ds.tree[split][indices][:, cols]
        y[offset:stop] = ds.common[split]["relevance"][indices]
        all_groups.append(groups)
        offset = stop
    final_groups = np.concatenate(all_groups).astype(np.int32)
    assert int(final_groups.sum()) == total_rows
    return X, y, final_groups


def train_ranker(ds: Dataset, segments, boost_rounds: int, seed: int = 42, cols: np.ndarray | None = None):
    assert lgb is not None, "lightgbm 不可用"
    X, y, groups = build_training_arrays(ds, segments, TRAIN_STOCK_CAP, cols=cols)
    params = dict(LGB_PARAMS)
    params.update({"seed": int(seed), "feature_fraction_seed": int(seed), "bagging_seed": int(seed)})
    dataset = lgb.Dataset(X, label=y, group=groups, free_raw_data=True)
    started = time.time()
    model = lgb.train(params, dataset, num_boost_round=int(boost_rounds), callbacks=[lgb.log_evaluation(0)])
    elapsed = time.time() - started
    info = {"training_seconds": elapsed, "train_rows": int(X.shape[0]), "seed": int(seed), "rounds": int(boost_rounds)}
    del dataset, X, y, groups
    gc.collect()
    return model, info


def predict_interval(ds: Dataset, model, split: str, start_time: int, stop_time: int, num_iteration: int, cols: np.ndarray | None = None, chunk_size: int = 250_000):
    if cols is None:
        cols = np.arange(FEATURE_STOP, dtype=np.int64)
    rows, groups = row_slice_for_times(ds, split, start_time, stop_time)
    prediction = np.empty(int(rows.stop) - int(rows.start), dtype=np.float32)
    for begin in range(int(rows.start), int(rows.stop), chunk_size):
        end = min(begin + chunk_size, int(rows.stop))
        prediction[begin - int(rows.start):end - int(rows.start)] = model.predict(
            ds.tree[split][begin:end][:, cols], num_iteration=int(num_iteration)
        ).astype(np.float32)
    return prediction, groups


def interval_target(ds: Dataset, split: str, start_time: int, stop_time: int) -> np.ndarray:
    rows, _ = row_slice_for_times(ds, split, start_time, stop_time)
    return np.asarray(ds.common[split]["y"][rows], dtype=np.float32)


# ---------------------------------------------------------------------------
# 预测指纹缓存
# ---------------------------------------------------------------------------
def model_fingerprint(model_type: str, segments, rounds: int, seed: int, config_name: str, params_extra: dict | None = None) -> str:
    payload = {
        "model_type": model_type,
        "segments": [[s, int(a), int(b)] for s, a, b in segments],
        "rounds": int(rounds),
        "seed": int(seed),
        "feature_config": config_name,
        "params": params_extra or {},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def get_cached_prediction(cache_dir: Path, key: str, expected_size: int) -> np.ndarray | None:
    path = cache_dir / f"{key}.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    if arr.shape[0] != expected_size:
        return None
    return arr


def save_cached_prediction(cache_dir: Path, key: str, array: np.ndarray) -> None:
    atomic_save_npy(cache_dir / f"{key}.npy", array)


# ---------------------------------------------------------------------------
# 融合与网格
# ---------------------------------------------------------------------------
def blend_rank_vectors(anchor_raw: np.ndarray, expert_raw: np.ndarray, groups: np.ndarray, weight: float) -> np.ndarray:
    """每个时间截面：先百分位秩，再加权，最后再次百分位秩。"""
    anchor_rank = group_rank_transform(anchor_raw, groups)
    expert_rank = group_rank_transform(expert_raw, groups)
    combined = (1.0 - weight) * anchor_rank + weight * expert_rank
    return group_rank_transform(combined, groups)


def vector_to_grid(prediction: np.ndarray, split: str, ds: Dataset, time_start: int, time_points: int,
                   interval: tuple[int, int] | None = None) -> np.ndarray:
    grid = np.full((time_points, S), 0.5, dtype=np.float32)
    if interval is None:
        times = np.asarray(ds.common[split]["time"], dtype=np.int32)
        stocks = np.asarray(ds.common[split]["stock"], dtype=np.int32)
    else:
        rows, _ = row_slice_for_times(ds, split, interval[0], interval[1])
        times = np.asarray(ds.common[split]["time"][rows], dtype=np.int32)
        stocks = np.asarray(ds.common[split]["stock"][rows], dtype=np.int32)
    grid[times - time_start, stocks] = prediction
    return grid


def mean_cross_sectional_rank_correlation(left_grid, right_grid, groups, stocks) -> float:
    values = []
    offset = 0
    for local_time, size in enumerate(groups):
        size = int(size)
        current_stocks = np.asarray(stocks[offset:offset + size], dtype=np.int32)
        values.append(rank_ic(left_grid[local_time, current_stocks], right_grid[local_time, current_stocks]))
        offset += size
    assert offset == stocks.size
    return float(np.nanmean(values))


def validate_prediction_grid(grid: np.ndarray, eval_mask: np.ndarray) -> dict:
    """校验预测网格。eval_mask 必须是显式评价位置掩码（不能用 !=0.5 启发式，
    因为偶数股票组的中间百分位秩可能恰好等于 0.5）。"""
    check = {
        "shape": list(grid.shape),
        "dtype": str(grid.dtype),
        "finite": bool(np.isfinite(grid).all()),
        "evaluation_count": int(np.count_nonzero(eval_mask)),
        "non_evaluation_count": int(np.count_nonzero(~eval_mask)),
        "non_evaluation_all_0_5": bool(np.all(grid[~eval_mask] == 0.5)),
        "minimum": float(grid.min()),
        "maximum": float(grid.max()),
        "mean": float(grid.mean()),
        "std": float(grid.std()),
    }
    assert grid.shape == (TEST_TIME_POINTS, S), grid.shape
    assert grid.dtype == np.float32
    assert np.isfinite(grid).all()
    assert check["evaluation_count"] == int(np.count_nonzero(eval_mask))
    assert check["non_evaluation_all_0_5"]
    return check
