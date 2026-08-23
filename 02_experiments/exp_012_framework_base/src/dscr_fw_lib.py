"""exp_012 统一运行时库（framework base）。

以 exp_011 已验证的实现（dscr_exp011_lib.py）为唯一事实底座，在其上增加框架级能力：
- 标准折叠定义（开发折 + 影子验证 + 官方 Valid 一次性检查）
- 决策种子集（与 exp_011 对齐）
- DecisionLog：预注册 / 自动比对 / 判定归档（对抗选择过拟合的唯一事实来源）
- 晋级门槛自动求值（与 exp_009/010/011 阈值完全一致）
- 多级预测缓存（primary + fallback，指纹一致时零成本复用）

禁止在本库中重新实现 RankIC / 抽样 / 训练 / 融合等核心逻辑；一律复用 exp_011 底座。
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

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
if os.environ.get("DSCR_FW_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["DSCR_FW_PROJECT_ROOT"]).resolve()

EXP011_SRC = PROJECT_ROOT / "02_experiments" / "exp_011_stable_anchor_retrain" / "src"
import sys  # noqa: E402

sys.path.insert(0, str(EXP011_SRC))

from dscr_exp011_lib import (  # noqa: E402,F401  (proven base, re-exported)
    ABLATION_CONFIGS, ANCHOR_EXPECTED_VALID_IC, ANCHOR_REPRO_TOLERANCE, BASE_ROUNDS,
    Dataset, FEATURE_BLOCKS, FEATURE_STOP, LGB_PARAMS, RECENT_LOOKBACK, RECENT_ROUNDS,
    S, SEEDS, TEST_START, TEST_STOP, TEST_TIME_POINTS, TRAIN_START, TREE_COUNT,
    TRAIN_STOCK_CAP, VALID_START, VALID_STOP, WEIGHT_CANDIDATES,
    atomic_save_npy, atomic_write_csv, atomic_write_json, atomic_write_text,
    blend_rank_vectors, build_training_arrays, capped_indices_for_split,
    feature_cols_for_config, file_sha256, get_cached_prediction, group_rank_ic_series, group_rank_transform, interval_target,
    json_ready, mean_cross_sectional_rank_correlation, model_fingerprint,
    predict_interval, rank_ic_self_test, row_slice_for_times, save_cached_prediction,
    score_prediction, stratified_capped_indices, train_ranker,
    validate_prediction_grid, vector_to_grid,
)

try:
    import lightgbm as lgb  # noqa: F401
except Exception:  # pragma: no cover
    lgb = None

# ---------------------------------------------------------------------------
# 框架级常量
# ---------------------------------------------------------------------------
# 决策种子集：前三个与 exp_011 对齐（可跨实验直接比较），后两个为候选复核种子
DECISION_SEEDS = (42, 2026, 3407)
CANDIDATE_SEEDS = (12345, 777)
ALL_SEEDS = DECISION_SEEDS + CANDIDATE_SEEDS

# 标准折叠：训练区间 -> 预测区间（与 exp_009/011 完全一致，保证跨实验可比）
FOLD_SPECS = {
    "fold_1": {"train": (TRAIN_START, 2189), "predict": (2189, 2432)},
    "fold_2": {"train": (TRAIN_START, 2432), "predict": (2432, 2675)},
    "shadow": {"train": (TRAIN_START, 2675), "predict": (2675, 2918)},
}
DEV_FOLDS = ("fold_1", "fold_2")
RECENT_DEV_TRAIN_START = {  # max(TRAIN_START, predict_start - RECENT_LOOKBACK)
    "fold_1": max(TRAIN_START, 2189 - RECENT_LOOKBACK),
    "fold_2": max(TRAIN_START, 2432 - RECENT_LOOKBACK),
    "shadow": max(TRAIN_START, 2675 - RECENT_LOOKBACK),
}

# 晋级门槛（阈值全部来自 exp_009/010/011 的既有门槛，不可事后修改）
GATE_SPECS = {
    "anchor_reproduction": {"max_abs_delta": ANCHOR_REPRO_TOLERANCE},
    "dev_fold_mean_delta": {"min": 0.0005},
    "dev_fold_all_positive": {},
    "dev_worst_fold_delta": {"min": 0.0},
    "shadow_delta": {"min": 0.0},
    "shadow_second_half_delta": {"min": -0.0005},
    "official_valid_mean_delta": {"min": 0.0003},
    "official_valid_late_delta": {"min": -0.0002},
    "official_valid_worst_quarter_delta": {"min": -0.0015},
    "plateau_span": {"max": 0.001},
    "test_anchor_rank_corr": {"min": 0.98},
    "seed_ensemble_stable": {},
}

# 转移衰减登记表（从历史实验直接填充；用于阶段 2/3 的迁移预算）
# 注：valid_delta 为官方 Valid 一次性检查增量；decay_rate = valid_delta / fold_delta。
# exp_012_fusion 线上增量（相对 exp_003 正式提交）= +0.000160，进一步显示本地 Valid 高估绝对增益。
TRANSFER_TABLE = [
    {"candidate": "exp_009 recent16 w0.25", "fold_delta": 0.001057, "valid_delta": 0.000675},
    {"candidate": "exp_011 recent16 w0.30", "fold_delta": 0.001022, "valid_delta": 0.000694},
    {"candidate": "exp_010 enhanced block", "fold_delta": 0.000890, "valid_delta": 0.000000},
    {"candidate": "exp_008 full system", "fold_delta": 0.006000, "valid_delta": -0.008555},
    {"candidate": "exp_012 catboost expert", "fold_delta": 0.007512, "valid_delta": 0.004341},
    {"candidate": "exp_012 cat337 block", "fold_delta": 0.001307, "valid_delta": 0.000754},
]
for _row in TRANSFER_TABLE:
    _row["decay_rate"] = _row["valid_delta"] / _row["fold_delta"] if _row["fold_delta"] != 0 else 0.0


def transfer_decay_rate(candidate: str) -> float | None:
    for row in TRANSFER_TABLE:
        if row["candidate"] == candidate:
            return row["decay_rate"]
    return None


# ---------------------------------------------------------------------------
# 多级预测缓存（primary 写、fallback 读并复制）
# ---------------------------------------------------------------------------
class PredictionCache:
    """统一预测缓存。get 时按 primary -> fallback 顺序查找，命中 fallback 后复制到 primary。

    指纹（model_type + segments + rounds + seed + config + 区间）与 exp_011 完全一致，
    因此可零成本复用 exp_011 runtime_cache 中已验证的预测。
    """

    def __init__(self, primary_dir: Path, fallback_dirs: list[Path] | None = None):
        self.primary = Path(primary_dir)
        self.primary.mkdir(parents=True, exist_ok=True)
        self.fallbacks = [Path(p) for p in (fallback_dirs or [])]

    def key(self, model_type, segments, rounds, seed, config_name, split, start, stop) -> str:
        base = model_fingerprint(model_type, segments, rounds, seed, config_name)
        cols = feature_cols_for_config(config_name)
        cols_hash = hashlib.sha256(np.ascontiguousarray(cols).tobytes()).hexdigest()[:8]
        return f"{base}_{cols_hash}_{split}_{int(start)}_{int(stop)}"

    def get(self, ds, model_type, segments, rounds, seed, config_name, split, start, stop):
        """返回 (pred, groups, info)。缓存缺失时训练并保存到 primary。"""
        key = self.key(model_type, segments, rounds, seed, config_name, split, start, stop)
        rows, groups = row_slice_for_times(ds, split, start, stop)
        expected = int(rows.stop) - int(rows.start)
        cached = get_cached_prediction(self.primary, key, expected)
        if cached is None:
            for fb in self.fallbacks:
                cached = get_cached_prediction(fb, key, expected)
                if cached is not None:
                    save_cached_prediction(self.primary, key, cached)
                    break
        if cached is not None:
            return cached, groups, None
        cols = feature_cols_for_config(config_name)
        model, info = train_ranker(ds, segments, rounds, seed=seed, cols=cols)
        pred, groups = predict_interval(ds, model, split, start, stop, rounds, cols=cols)
        save_cached_prediction(self.primary, key, pred)
        return pred, groups, info

    def get_animal(self, ds, model_type, segments, rounds, seed, cols_name,
                   framework, split, start, stop, params_extra: dict | None = None):
        """动物专用：framework 为 'lgbm_lambdarank' / 'lgbm_xendcg' / 'catboost'。

        指纹包含 framework 与 cols_name，避免不同实现共用缓存键。
        """
        cols = COLS_CONFIGS[cols_name]
        payload = {"model_type": model_type, "segments": [[s, int(a), int(b)] for s, a, b in segments],
                   "rounds": int(rounds), "seed": int(seed), "feature_config": cols_name,
                   "framework": framework, "params": params_extra or {}}
        base = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        cols_hash = hashlib.sha256(np.ascontiguousarray(cols).tobytes()).hexdigest()[:8]
        key = f"{base}_{cols_hash}_{split}_{int(start)}_{int(stop)}"
        rows, groups = row_slice_for_times(ds, split, start, stop)
        expected = int(rows.stop) - int(rows.start)
        cached = get_cached_prediction(self.primary, key, expected)
        if cached is None:
            for fb in self.fallbacks:
                cached = get_cached_prediction(fb, key, expected)
                if cached is not None:
                    save_cached_prediction(self.primary, key, cached)
                    break
        if cached is not None:
            return cached, groups, None
        if framework == "catboost":
            model, info = train_catboost_ranker(ds, segments, rounds, seed=seed, cols=cols,
                                                params_extra=params_extra)
            pred, groups = predict_interval_catboost(ds, model, split, start, stop, cols=cols)
        elif framework in ("lgbm_lambdarank", "lgbm_xendcg"):
            objective = "lambdarank" if framework == "lgbm_lambdarank" else "rank_xendcg"
            model, info = train_lgbm_objective(ds, segments, rounds, seed=seed, cols=cols,
                                               objective=objective)
            pred, groups = predict_interval(ds, model, split, start, stop, rounds, cols=cols)
        else:
            raise ValueError(f"未知框架: {framework}")
        save_cached_prediction(self.primary, key, pred)
        return pred, groups, info


# ---------------------------------------------------------------------------
# DecisionLog：预注册 + 自动比对 + 判定归档
# ---------------------------------------------------------------------------
class DecisionLog:
    """每个候选一条记录。运行前必须 pre_register（冻结门槛与参数），运行后 verify。

    任何"看结果改门槛"的行为都会破坏 pre-registration 指纹，从而在判定时显式暴露。
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def pre_register(self, experiment_id: str, candidate_id: str, params: dict,
                     gates_expected: dict | None = None) -> str:
        record_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{candidate_id}"
        entry = {
            "record_id": record_id,
            "experiment_id": experiment_id,
            "candidate_id": candidate_id,
            "pre_registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": json_ready(params),
            "gates_expected": json_ready(gates_expected or GATE_SPECS),
            "status": "pre_registered",
            "results": None,
            "verdict": None,
            "verified_at": None,
        }
        atomic_write_json(self.log_dir / f"{record_id}.json", entry)
        return record_id

    def verify(self, record_id: str, results: dict) -> dict:
        path = self.log_dir / f"{record_id}.json"
        assert path.exists(), f"决策记录缺失: {record_id}"
        entry = json.loads(path.read_text(encoding="utf-8"))
        expected = entry["gates_expected"]
        outcomes = {}
        all_pass = True
        for gate, spec in expected.items():
            if gate not in results:
                outcomes[gate] = {"measured": None, "pass": False, "reason": "missing_measurement"}
                all_pass = False
                continue
            measured = results[gate]
            ok = True
            reasons = []
            if "max_abs_delta" in spec:
                ok &= bool(abs(measured) <= spec["max_abs_delta"])
                reasons.append(f"|Δ|={abs(measured):.6f}≤{spec['max_abs_delta']}")
            if "min" in spec:
                ok &= bool(measured >= spec["min"])
                reasons.append(f"{measured:.6f}≥{spec['min']}")
            if "max" in spec:
                ok &= bool(measured <= spec["max"])
                reasons.append(f"{measured:.6f}≤{spec['max']}")
            outcomes[gate] = {"measured": measured, "pass": bool(ok), "detail": "; ".join(reasons)}
            all_pass &= bool(ok)
        entry["results"] = json_ready(results)
        entry["status"] = "verified"
        entry["verdict"] = "promoted" if all_pass else "not_promoted"
        entry["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        entry["gates_outcomes"] = outcomes
        atomic_write_json(path, entry)
        return {"record_id": record_id, "verdict": entry["verdict"], "outcomes": outcomes}


# ---------------------------------------------------------------------------
# 晋级门槛求值（单一入口，输入指标即输出全部门槛测量值）
# ---------------------------------------------------------------------------
def evaluate_gates(anchor_repro_delta: float, dev_fold_deltas: list[float],
                   shadow_blend_delta: float, shadow_blend_late_delta: float,
                   valid_anchor_metrics: dict, valid_blend_metrics: dict,
                   plateau_span: float, test_anchor_corr: float,
                   seed_ensemble_stable: bool) -> dict:
    """按 GATE_SPECS 计算全部门槛测量值（阈值比对交给 DecisionLog.verify）。"""
    return {
        "anchor_reproduction": float(anchor_repro_delta),
        "dev_fold_mean_delta": float(np.mean(dev_fold_deltas)),
        "dev_fold_all_positive": bool(all(d > 0 for d in dev_fold_deltas)),
        "dev_worst_fold_delta": float(min(dev_fold_deltas)),
        "shadow_delta": float(shadow_blend_delta),
        "shadow_second_half_delta": float(shadow_blend_late_delta),
        "official_valid_mean_delta": float(valid_blend_metrics["mean_rankic"] - valid_anchor_metrics["mean_rankic"]),
        "official_valid_late_delta": float(valid_blend_metrics["second_half_rankic"] - valid_anchor_metrics["second_half_rankic"]),
        "official_valid_worst_quarter_delta": float(valid_blend_metrics["worst_quarter_rankic"] - valid_anchor_metrics["worst_quarter_rankic"]),
        "plateau_span": float(plateau_span),
        "test_anchor_rank_corr": float(test_anchor_corr),
        "seed_ensemble_stable": bool(seed_ensemble_stable),
    }


# ---------------------------------------------------------------------------
# 多种子集成
# ---------------------------------------------------------------------------
def seed_ensemble_preds(per_seed_preds: list[np.ndarray], groups: np.ndarray) -> np.ndarray:
    ranks = [group_rank_transform(p, groups) for p in per_seed_preds]
    return np.mean(ranks, axis=0).astype(np.float32)


def segment_anchor(stop_time: int) -> list[tuple[str, int, int]]:
    return [("train", TRAIN_START, stop_time)]


def segment_recent(stop_time: int) -> list[tuple[str, int, int]]:
    return [("train", max(TRAIN_START, stop_time - RECENT_LOOKBACK), stop_time)]


# ---------------------------------------------------------------------------
# 模型动物园：列视图配置
# ---------------------------------------------------------------------------
COLS_CONFIGS = {
    # legacy_328（默认）
    "legacy_328": np.arange(FEATURE_STOP, dtype=np.int64),
    # legacy_328 + 9 个类别列（tree 视图列 408-416；类别感知候选）
    "legacy_328_cats": np.sort(np.concatenate([
        np.arange(0, FEATURE_STOP, dtype=np.int64),
        np.arange(408, 417, dtype=np.int64),
    ])),
    # 全 tree 419 视图（对照 exp_005 的历史结论）
    "tree_419": np.arange(419, dtype=np.int64),
}
COLS_CONFIGS["legacy_328"].flags.writeable = False
COLS_CONFIGS["legacy_328_cats"].flags.writeable = False
COLS_CONFIGS["tree_419"].flags.writeable = False


# ---------------------------------------------------------------------------
# CatBoost 排序器（第二棵 GBDT 正交性候选）
# ---------------------------------------------------------------------------
try:
    import catboost as cb  # noqa: F401
except Exception:  # pragma: no cover
    cb = None

CATBOOST_PARAMS = {
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.65,
    "loss_function": "YetiRank",
    "task_type": "CPU",
    "thread_count": max(1, (os.cpu_count() or 8) - 2),
    "verbose": False,
    "allow_writing_files": False,
}


def train_catboost_ranker(ds: Dataset, segments, iterations: int, seed: int = 42,
                          cols: np.ndarray | None = None, params_extra: dict | None = None):
    assert cb is not None, "catboost 不可用"
    if cols is None:
        cols = COLS_CONFIGS["legacy_328"]
    X, y, groups = build_training_arrays(ds, segments, TRAIN_STOCK_CAP, cols=cols)
    group_ids = np.repeat(np.arange(groups.size, dtype=np.int64), groups.astype(np.int64))
    params = dict(CATBOOST_PARAMS)
    if params_extra:
        params.update(params_extra)
    params["random_seed"] = int(seed)
    pool = cb.Pool(X, label=y, group_id=group_ids)
    started = time.time()
    model = cb.train(pool, iterations=int(iterations), params=params)
    elapsed = time.time() - started
    info = {"training_seconds": elapsed, "train_rows": int(X.shape[0]), "seed": int(seed),
            "iterations": int(iterations), "framework": "catboost"}
    del pool, X, y, groups
    gc.collect()
    return model, info


def predict_interval_catboost(ds: Dataset, model, split: str, start_time: int, stop_time: int,
                              cols: np.ndarray | None = None, chunk_size: int = 250_000):
    if cols is None:
        cols = COLS_CONFIGS["legacy_328"]
    rows, groups = row_slice_for_times(ds, split, start_time, stop_time)
    prediction = np.empty(int(rows.stop) - int(rows.start), dtype=np.float32)
    for begin in range(int(rows.start), int(rows.stop), chunk_size):
        end = min(begin + chunk_size, int(rows.stop))
        prediction[begin - int(rows.start):end - int(rows.start)] = model.predict(
            ds.tree[split][begin:end][:, cols]).astype(np.float32)
    return prediction, groups


def train_lgbm_objective(ds: Dataset, segments, rounds: int, seed: int = 42,
                         cols: np.ndarray | None = None, objective: str = "rank_xendcg"):
    """LightGBM 变体目标函数（如 rank_xendcg）。"""
    assert lgb is not None, "lightgbm 不可用"
    if cols is None:
        cols = COLS_CONFIGS["legacy_328"]
    X, y, groups = build_training_arrays(ds, segments, TRAIN_STOCK_CAP, cols=cols)
    params = dict(LGB_PARAMS)
    params.update({"objective": objective, "seed": int(seed),
                   "feature_fraction_seed": int(seed), "bagging_seed": int(seed)})
    dataset = lgb.Dataset(X, label=y, group=groups, free_raw_data=True)
    started = time.time()
    model = lgb.train(params, dataset, num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(0)])
    elapsed = time.time() - started
    info = {"training_seconds": elapsed, "train_rows": int(X.shape[0]), "seed": int(seed),
            "rounds": int(rounds), "objective": objective}
    del dataset, X, y, groups
    gc.collect()
    return model, info
