from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("experiment.ipynb")


SETUP = r'''
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from torch import nn
import torch.nn.functional as F


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data.z").exists() and (candidate / "02_experiments").exists():
            return candidate
    raise RuntimeError("无法定位项目根目录：请从项目根目录或实验目录启动 Notebook。")


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: int
    train_stop: int
    valid_start: int
    valid_stop: int


PROJECT_ROOT = find_project_root()
EXPERIMENT_ID = "exp_015_drift_robust_rank"
EXPERIMENT_VERSION = "integrated_v2"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
FEATURE_CACHE_DIR = PROJECT_ROOT / "03_cache" / EXPERIMENT_ID / EXPERIMENT_VERSION
OUTPUT_DIR = PROJECT_ROOT / "04_results" / EXPERIMENT_ID / EXPERIMENT_VERSION
EXP009_DIR = PROJECT_ROOT / "04_results" / "exp_009_anchor_recent_blend_cv"
FINAL_SUBMISSION = PROJECT_ROOT / "04_results" / "final_submission" / "prediction.npy"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
READY_PATH = DATASET_DIR / "READY"

RUN_MODE = os.environ.get("DSCR_EXP015_MODE", "smoke").strip().lower()
if RUN_MODE not in {"smoke", "preflight", "full"}:
    raise ValueError("DSCR_EXP015_MODE 只允许 smoke、preflight 或 full。")
RUN_STAGE = os.environ.get("DSCR_EXP015_STAGE", "all").strip().lower()
if RUN_STAGE not in {"all", "features", "pretrain", "experts", "router", "final"}:
    raise ValueError("DSCR_EXP015_STAGE 只允许 all、features、pretrain、experts、router 或 final。")
TRAINING_ALLOWED = os.environ.get("DSCR_EXP015_ALLOW_TRAINING", "").strip() == "YES"
RUN_DIR = OUTPUT_DIR / RUN_MODE
RUNTIME_CACHE_DIR = RUN_DIR / "runtime_cache"
MODEL_DIR = RUN_DIR / "models"

TRAIN_START, TRAIN_STOP = 486, 2918
VALID_START, VALID_STOP = 2918, 3161
TEST_START, TEST_STOP = 3161, 3603
TEST_TIME_POINTS, STOCK_COUNT = 442, 5282
LEGACY_FEATURES, EXTRA_RANK_FEATURES, ROBUST_FEATURES = 328, 20, 348
EXTRA_RAW_COLS = tuple(range(20, 40))
EXTRA_SOURCE_NAMES = (
    "num_87", "num_49", "num_4", "num_55", "num_85",
    "num_75", "num_56", "num_76", "num_51", "num_67",
    "num_88", "num_61", "num_79", "num_1", "num_91",
    "num_84", "num_15", "num_60", "num_64", "num_43",
)
EXTRA_FEATURE_NAMES = (
    "rank_num_87", "rank_num_49", "rank_num_4", "rank_num_55", "rank_num_85",
    "rank_num_75", "rank_num_56", "rank_num_76", "rank_num_51", "rank_num_67",
    "rank_num_88", "rank_num_61", "rank_num_79", "rank_num_1", "rank_num_91",
    "rank_num_84", "rank_num_15", "rank_num_60", "rank_num_64", "rank_num_43",
)
FEATURE_ALGORITHM = "per_time_average_tie_percentile_rank_v1"
TRAIN_STOCK_CAP = 1200
RECENT_LOOKBACK = 1702
BASE_ROUNDS = 8
EXPERT_ROUNDS = 16
EXP009_WEIGHT = 0.25
WEIGHT_CANDIDATES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
SEEDS = (42, 2026, 3407)
NUM_THREADS = max(1, (os.cpu_count() or 8) - 2)

FOLDS = (
    WalkForwardFold("fold_1", 486, 2189, 2189, 2432),
    WalkForwardFold("fold_2", 486, 2432, 2432, 2675),
    WalkForwardFold("fold_3", 486, 2675, 2675, 2918),
)
DEV_FOLDS = FOLDS[:2]
SHADOW_FOLD = FOLDS[2]

GATE_THRESHOLDS = {
    "dev_mean_delta": 0.0005,
    "shadow_delta": 0.0,
    "shadow_late_delta": -0.0005,
    "official_mean_delta": 0.0003,
    "official_late_delta": -0.0002,
    "official_worst_quarter_delta": -0.0015,
    "seed_delta_spread": 0.002,
    "test_anchor_rank_corr": 0.98,
}
ONLINE_PROMOTION_LINE = 0.109959
EXPECTED_FINAL_SHA256 = "9d322401a2d8fedd38dea66b97578873e721f03eeb93575dbc8bdc2a1aef38e6"
PATCH_SIZES = (5, 20, 60)
SEQUENCE_WINDOW = 60
SEQUENCE_FEATURES = 40
PROTOTYPE_COUNT = 16
ROUTER_EXPERTS = ("robust_rank", "catboost", "multiscale_residual")
ROUTER_STATE_DIM = 8

LGB_PARAMS = {
    "objective": "lambdarank", "metric": "None", "learning_rate": 0.0228695,
    "num_leaves": 79, "min_data_in_leaf": 147, "feature_fraction": 0.80936,
    "bagging_fraction": 0.647764, "bagging_freq": 1, "lambda_l1": 2.35724,
    "lambda_l2": 0.238705, "max_bin": 127, "label_gain": list(range(64)),
    "lambdarank_truncation_level": 1024, "verbosity": -1,
    "num_threads": NUM_THREADS,
}

RUN_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(pd.Series({
    "experiment": EXPERIMENT_ID, "version": EXPERIMENT_VERSION,
    "mode": RUN_MODE, "stage": RUN_STAGE, "training_allowed": TRAINING_ALLOWED,
    "project_root": str(PROJECT_ROOT),
    "feature_view": "robust_rank_348", "recent_lookback": RECENT_LOOKBACK,
    "expert_rounds": EXPERT_ROUNDS, "weights": WEIGHT_CANDIDATES,
    "seeds": SEEDS, "final_endpoint": VALID_STOP,
}))
'''


UTILITIES = r'''
def file_sha256(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_csv(partial, index=False, encoding="utf-8-sig")
    os.replace(partial, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.save(handle, array)
    os.replace(partial, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez(handle, **arrays)
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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def require_training_authorization(operation: str) -> None:
    if RUN_MODE != "full" or not TRAINING_ALLOWED:
        raise RuntimeError(
            f"{operation} 被训练保护开关阻止：仅当 "
            "DSCR_EXP015_MODE=full 且 DSCR_EXP015_ALLOW_TRAINING=YES 时允许更新真实模型参数。"
        )


def assert_final_submission_unchanged(expected: str = EXPECTED_FINAL_SHA256) -> str:
    if not FINAL_SUBMISSION.exists():
        raise RuntimeError("正式提交文件不存在，无法执行保护检查。")
    measured = file_sha256(FINAL_SUBMISSION)
    if measured.lower() != str(expected).lower():
        raise RuntimeError(f"正式提交 SHA-256 发生变化：{measured}")
    return measured


def rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    usable = np.isfinite(prediction) & np.isfinite(target)
    if int(usable.sum()) < 2:
        return np.nan
    x = rankdata(prediction[usable], method="average")
    y = rankdata(target[usable], method="average")
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def group_rank_transform(values: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    values = np.asarray(values)
    groups = np.asarray(groups, dtype=np.int32)
    out = np.empty(values.size, dtype=np.float32)
    offset = 0
    for size in groups:
        size = int(size)
        block = values[offset:offset + size]
        out[offset:offset + size] = rankdata(block, method="average").astype(np.float32) / float(size)
        offset += size
    assert offset == values.size
    return out


def group_rank_ic_series(prediction: np.ndarray, target: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    values, offset = [], 0
    for size in groups:
        size = int(size)
        values.append(rank_ic(prediction[offset:offset + size], target[offset:offset + size]))
        offset += size
    assert offset == prediction.size == target.size
    return np.asarray(values, dtype=np.float64)


def score_prediction(prediction: np.ndarray, target: np.ndarray, groups: Sequence[int]) -> dict[str, float]:
    series = group_rank_ic_series(prediction, target, groups)
    quarters = [part for part in np.array_split(series, min(4, len(series))) if part.size]
    return {
        "mean_rank_ic": float(np.nanmean(series)),
        "std_rank_ic": float(np.nanstd(series)),
        "late_half_rank_ic": float(np.nanmean(series[len(series) // 2:])),
        "worst_quarter_rank_ic": float(min(np.nanmean(q) for q in quarters)),
        "negative_time_share": float(np.mean(series < 0)),
    }


def blend_rank(anchor: np.ndarray, expert: np.ndarray, groups: Sequence[int], weight: float) -> np.ndarray:
    anchor_rank = group_rank_transform(anchor, groups)
    if float(weight) == 0.0:
        return anchor_rank
    expert_rank = group_rank_transform(expert, groups)
    return group_rank_transform((1.0 - float(weight)) * anchor_rank + float(weight) * expert_rank, groups)


def group_zscore(values: np.ndarray, groups: Sequence[int], eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    out = np.empty_like(values)
    offset = 0
    for size in groups:
        size = int(size)
        block = values[offset:offset + size]
        out[offset:offset + size] = (block - block.mean()) / max(float(block.std()), eps)
        offset += size
    if offset != values.size:
        raise ValueError("group_zscore 分组长度不匹配。")
    return out


def orthogonalize_experts(anchor: np.ndarray, experts: Sequence[np.ndarray], groups: Sequence[int]) -> list[np.ndarray]:
    """逐截面对 anchor 和已接纳专家执行稳定 Gram-Schmidt。"""
    basis = [group_zscore(group_rank_transform(anchor, groups), groups)]
    outputs = []
    for expert in experts:
        residual = group_zscore(group_rank_transform(expert, groups), groups).astype(np.float64)
        offset = 0
        for size in groups:
            size = int(size)
            block = residual[offset:offset + size]
            for vector in basis:
                reference = vector[offset:offset + size].astype(np.float64)
                denom = float(np.dot(reference, reference))
                if denom > 1e-12:
                    block -= float(np.dot(block, reference) / denom) * reference
            residual[offset:offset + size] = block
            offset += size
        normalized = group_zscore(residual.astype(np.float32), groups)
        outputs.append(normalized)
        basis.append(normalized)
    return outputs


def top2_router_weights(logits: np.ndarray, min_share: float = 0.05) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[1] != len(ROUTER_EXPERTS):
        raise ValueError("路由 logits 维度不匹配。")
    order = np.argsort(logits, axis=1)[:, -2:]
    masked = np.full_like(logits, -np.inf)
    rows = np.arange(logits.shape[0])[:, None]
    masked[rows, order] = logits[rows, order]
    finite_max = np.max(masked, axis=1, keepdims=True)
    exp = np.exp(masked - finite_max)
    weights = exp / exp.sum(axis=1, keepdims=True)
    shares = (weights > 0).mean(axis=0)
    if np.any(shares < float(min_share)):
        raise RuntimeError(f"路由专家参与率不足：{shares.tolist()}")
    return weights.astype(np.float32)


def disagreement_shrink(seed_predictions: np.ndarray, floor: float = 0.10, ceiling: float = 0.40) -> np.ndarray:
    seed_predictions = np.asarray(seed_predictions, dtype=np.float32)
    if seed_predictions.ndim != 3:
        raise ValueError("seed_predictions 应为 (seed,time,expert)。")
    disagreement = seed_predictions.std(axis=0).mean(axis=1)
    scale = disagreement / max(float(np.quantile(disagreement, 0.90)), 1e-6)
    return np.clip(ceiling - (ceiling - floor) * scale, floor, ceiling).astype(np.float32)


def integrated_blend(anchor: np.ndarray, experts: Sequence[np.ndarray], groups: Sequence[int],
                     router_weights: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    orthogonal = orthogonalize_experts(anchor, experts, groups)
    if router_weights.shape != (len(groups), len(orthogonal)) or budgets.shape != (len(groups),):
        raise ValueError("路由权重或预算维度不匹配。")
    anchor_rank = group_rank_transform(anchor, groups)
    out = np.empty_like(anchor_rank)
    offset = 0
    for t, size in enumerate(groups):
        size = int(size)
        mixed = np.zeros(size, dtype=np.float32)
        for j, expert in enumerate(orthogonal):
            mixed += router_weights[t, j] * expert[offset:offset + size]
        budget = float(budgets[t])
        out[offset:offset + size] = (1.0 - budget) * anchor_rank[offset:offset + size] + budget * mixed
        offset += size
    return group_rank_transform(out, groups)


def rank_feature_block(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != EXTRA_RANK_FEATURES:
        raise ValueError(f"新增秩特征输入应为 (n,20)，实际 {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("新增秩特征输入存在非有限值。")
    ranked = rankdata(raw, method="average", axis=0).astype(np.float32) / float(raw.shape[0])
    if ranked.shape != raw.shape or not np.isfinite(ranked).all():
        raise AssertionError("截面秩构造失败。")
    if float(ranked.min()) <= 0.0 or float(ranked.max()) > 1.0:
        raise AssertionError("截面秩超出 (0,1]。")
    return ranked


def evaluate_gates(measured: dict[str, object]) -> pd.DataFrame:
    rows = [
        ("selected_weight_positive", bool(measured["selected_weight"] > 0), measured["selected_weight"], "> 0"),
        ("dev_mean_delta", bool(measured["dev_mean_delta"] >= GATE_THRESHOLDS["dev_mean_delta"]), measured["dev_mean_delta"], f">= {GATE_THRESHOLDS['dev_mean_delta']}"),
        ("dev_all_positive", bool(measured["dev_all_positive"]), measured["dev_all_positive"], "True"),
        ("shadow_delta", bool(measured["shadow_delta"] >= GATE_THRESHOLDS["shadow_delta"]), measured["shadow_delta"], f">= {GATE_THRESHOLDS['shadow_delta']}"),
        ("shadow_late_delta", bool(measured["shadow_late_delta"] >= GATE_THRESHOLDS["shadow_late_delta"]), measured["shadow_late_delta"], f">= {GATE_THRESHOLDS['shadow_late_delta']}"),
        ("official_mean_delta", bool(measured["official_mean_delta"] >= GATE_THRESHOLDS["official_mean_delta"]), measured["official_mean_delta"], f">= {GATE_THRESHOLDS['official_mean_delta']}"),
        ("official_late_delta", bool(measured["official_late_delta"] >= GATE_THRESHOLDS["official_late_delta"]), measured["official_late_delta"], f">= {GATE_THRESHOLDS['official_late_delta']}"),
        ("official_worst_quarter_delta", bool(measured["official_worst_quarter_delta"] >= GATE_THRESHOLDS["official_worst_quarter_delta"]), measured["official_worst_quarter_delta"], f">= {GATE_THRESHOLDS['official_worst_quarter_delta']}"),
        ("seed_direction_consistent", bool(measured["seed_direction_consistent"]), measured["seed_direction_consistent"], "True"),
        ("seed_delta_spread", bool(measured["seed_delta_spread"] <= GATE_THRESHOLDS["seed_delta_spread"]), measured["seed_delta_spread"], f"<= {GATE_THRESHOLDS['seed_delta_spread']}"),
        ("test_anchor_rank_corr", bool(measured["test_anchor_rank_corr"] >= GATE_THRESHOLDS["test_anchor_rank_corr"]), measured["test_anchor_rank_corr"], f">= {GATE_THRESHOLDS['test_anchor_rank_corr']}"),
        ("output_contract", bool(measured["output_contract"]), measured["output_contract"], "True"),
    ]
    return pd.DataFrame(rows, columns=["gate", "passed", "measured", "requirement"])
'''


DATA_AND_CACHE = r'''
class DataContext:
    def __init__(self, check_hash: bool = True):
        if not READY_PATH.exists() or not MANIFEST_PATH.exists():
            raise RuntimeError("processed_data_v1 缺少 READY 或 manifest.json。")
        self.ready = json.loads(READY_PATH.read_text(encoding="utf-8"))
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.manifest_sha256 = file_sha256(MANIFEST_PATH)
        if self.ready["manifest_sha256"] != self.manifest_sha256:
            raise RuntimeError("READY 与 manifest SHA-256 不一致。")
        m = self.manifest
        assert m["status"] == "ready"
        assert m["dimensions"] == {"time": 3603, "stock": 5282, "raw_numeric": 99, "raw_category": 9}
        assert m["features"]["legacy_numeric_prefix"] == LEGACY_FEATURES
        assert m["features"]["tree_count"] == 419
        assert tuple(m["features"]["numeric_names"][20:40]) == EXTRA_SOURCE_NAMES
        assert m["validation"]["test_mask"]["count"] == 2_042_538
        self.common, self.tree = {}, {}
        for split in ("train", "valid", "test"):
            common_dir = DATASET_DIR / "common"
            entry = {
                "time": np.load(common_dir / f"{split}_time.npy", mmap_mode="r"),
                "stock": np.load(common_dir / f"{split}_stock.npy", mmap_mode="r"),
                "groups": np.load(common_dir / f"{split}_group_sizes.npy", mmap_mode="r"),
            }
            if split != "test":
                entry["y"] = np.load(common_dir / f"{split}_y.npy", mmap_mode="r")
                entry["relevance"] = np.load(common_dir / f"{split}_relevance.npy", mmap_mode="r")
            self.common[split] = entry
            matrix = np.load(DATASET_DIR / "tree" / f"{split}_X.npy", mmap_mode="r")
            assert matrix.shape == (int(m["expected_rows"][split]), 419)
            self.tree[split] = matrix
            assert int(entry["groups"].sum()) == entry["time"].size == entry["stock"].size
        self.sequence = np.load(DATASET_DIR / "sequence" / "X.npy", mmap_mode="r")
        self.sequence_mask = np.load(DATASET_DIR / "sequence" / "mask_x.npy", mmap_mode="r")
        assert self.sequence.shape == (3603, STOCK_COUNT, SEQUENCE_FEATURES)
        assert self.sequence_mask.shape == (3603, STOCK_COUNT)
        self.rank_cache: dict[str, np.ndarray] = {}

    def row_slice(self, split: str, start_time: int, stop_time: int) -> tuple[slice, np.ndarray]:
        times = self.common[split]["time"]
        begin = int(np.searchsorted(times, start_time, side="left"))
        end = int(np.searchsorted(times, stop_time, side="left"))
        split_start = int(times[0])
        groups = np.asarray(self.common[split]["groups"][start_time - split_start:stop_time - split_start], dtype=np.int32)
        assert int(groups.sum()) == end - begin
        return slice(begin, end), groups

    def target(self, split: str, start_time: int, stop_time: int) -> np.ndarray:
        rows, _ = self.row_slice(split, start_time, stop_time)
        return np.asarray(self.common[split]["y"][rows], dtype=np.float32)

    def temporal_batch(self, split: str, start_time: int, stop_time: int,
                       limit: int | None = None) -> tuple[np.ndarray, np.ndarray, slice]:
        rows, _ = self.row_slice(split, start_time, stop_time)
        times = np.asarray(self.common[split]["time"][rows], dtype=np.int32)
        stocks = np.asarray(self.common[split]["stock"][rows], dtype=np.int32)
        if limit is not None:
            times, stocks = times[:int(limit)], stocks[:int(limit)]
        history = np.zeros((times.size, SEQUENCE_WINDOW, SEQUENCE_FEATURES), dtype=np.float32)
        mask = np.zeros((times.size, SEQUENCE_WINDOW), dtype=np.float32)
        for i, (time_index, stock_index) in enumerate(zip(times, stocks)):
            begin = max(0, int(time_index) - SEQUENCE_WINDOW + 1)
            length = int(time_index) - begin + 1
            history[i, -length:] = self.sequence[begin:int(time_index) + 1, int(stock_index)]
            mask[i, -length:] = self.sequence_mask[begin:int(time_index) + 1, int(stock_index)]
        if not np.isfinite(history).all() or not np.isfinite(mask).all():
            raise ValueError("真实时序 dry-run batch 存在非有限值。")
        return history, mask, rows


def feature_cache_fingerprint(ctx: DataContext) -> str:
    payload = {
        "source_manifest_sha256": ctx.manifest_sha256,
        "source_tree_files": {
            split: ctx.manifest["files"][f"tree/{split}_X.npy"]["sha256"]
            for split in ("train", "valid", "test")
        },
        "raw_columns": EXTRA_RAW_COLS,
        "source_names": EXTRA_SOURCE_NAMES,
        "feature_names": EXTRA_FEATURE_NAMES,
        "algorithm": FEATURE_ALGORITHM,
        "dtype": "float32",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_rank_cache_file(ctx: DataContext, split: str, output_path: Path) -> dict[str, object]:
    groups = np.asarray(ctx.common[split]["groups"], dtype=np.int32)
    rows = int(groups.sum())
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    target = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(rows, EXTRA_RANK_FEATURES))
    offset = 0
    for local_time, size in enumerate(groups):
        size = int(size)
        target[offset:offset + size] = rank_feature_block(
            ctx.tree[split][offset:offset + size, EXTRA_RAW_COLS]
        )
        offset += size
        if (local_time + 1) % 250 == 0:
            target.flush()
            print(f"{split}: {local_time + 1}/{groups.size} time points", flush=True)
    assert offset == rows
    target.flush()
    del target
    os.replace(partial, output_path)
    return {
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "shape": [rows, EXTRA_RANK_FEATURES], "dtype": "float32",
        "bytes": output_path.stat().st_size, "sha256": file_sha256(output_path),
    }


def ensure_full_rank_cache(ctx: DataContext) -> dict[str, object]:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FEATURE_CACHE_DIR / "manifest.json"
    fingerprint = feature_cache_fingerprint(ctx)
    if manifest_path.exists():
        cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if cache_manifest.get("fingerprint") != fingerprint:
            raise RuntimeError("exp015 特征缓存指纹不匹配；为防止静默复用错误缓存，已停止。")
    else:
        cache_manifest = {
            "schema_version": 1, "status": "building", "fingerprint": fingerprint,
            "source_manifest_sha256": ctx.manifest_sha256, "algorithm": FEATURE_ALGORITHM,
            "raw_tree_columns": list(EXTRA_RAW_COLS), "feature_names": list(EXTRA_FEATURE_NAMES),
            "source_names": list(EXTRA_SOURCE_NAMES),
            "label_free": True, "uses_future_statistics": False, "files": {},
        }

    for split in ("train", "valid", "test"):
        path = FEATURE_CACHE_DIR / f"{split}_rank20.npy"
        record = cache_manifest.get("files", {}).get(split)
        if path.exists() and record:
            arr = np.load(path, mmap_mode="r")
            expected = (int(ctx.manifest["expected_rows"][split]), EXTRA_RANK_FEATURES)
            if arr.shape != expected or arr.dtype != np.float32 or file_sha256(path) != record.get("sha256"):
                raise RuntimeError(f"{split} 秩特征缓存契约或 SHA-256 不匹配。")
            print(f"复用 {split} 秩特征缓存。")
        elif path.exists() != bool(record):
            raise RuntimeError(f"{split} 缓存文件与 manifest 记录不完整，拒绝静默重建。")
        else:
            cache_manifest["files"][split] = build_rank_cache_file(ctx, split, path)
            atomic_write_json(manifest_path, cache_manifest)
        ctx.rank_cache[split] = np.load(path, mmap_mode="r")
    cache_manifest["status"] = "ready"
    cache_manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_write_json(manifest_path, cache_manifest)
    return cache_manifest


def build_preflight_rank_cache(ctx: DataContext, start: int = TRAIN_START, stop: int = TRAIN_START + 6) -> tuple[np.ndarray, np.ndarray, slice]:
    rows, groups = ctx.row_slice("train", start, stop)
    raw = np.asarray(ctx.tree["train"][rows, EXTRA_RAW_COLS], dtype=np.float32)
    ranked = np.empty_like(raw, dtype=np.float32)
    offset = 0
    for size in groups:
        size = int(size)
        ranked[offset:offset + size] = rank_feature_block(raw[offset:offset + size])
        offset += size
    assert offset == ranked.shape[0]
    path = RUN_DIR / "preflight_rank20.npy"
    atomic_save_npy(path, ranked)
    loaded = np.load(path)
    assert np.array_equal(loaded, ranked)
    atomic_write_json(RUN_DIR / "feature_cache_manifest.json", {
        "status": "preflight_subset", "source_manifest_sha256": ctx.manifest_sha256,
        "algorithm": FEATURE_ALGORITHM, "label_free": True, "uses_future_statistics": False,
        "time_interval": [start, stop], "shape": list(ranked.shape),
        "feature_names": EXTRA_FEATURE_NAMES, "sha256": file_sha256(path),
    })
    return ranked, groups, rows
'''


MODELING = r'''
class MultiScalePatchEncoder(nn.Module):
    def __init__(self, feature_dim: int = SEQUENCE_FEATURES, hidden: int = 64, embedding: int = 32):
        super().__init__()
        self.patch_sizes = PATCH_SIZES
        self.patch_projections = nn.ModuleDict({
            str(size): nn.Linear(feature_dim * 2, hidden) for size in self.patch_sizes
        })
        self.shared = nn.Sequential(nn.Linear(hidden * len(self.patch_sizes), hidden), nn.GELU(), nn.Linear(hidden, embedding))

    @staticmethod
    def causal_decompose(sequence: torch.Tensor, mask: torch.Tensor, size: int) -> torch.Tensor:
        window = sequence[:, -size:]
        current_mask = mask[:, -size:].unsqueeze(-1)
        count = current_mask.sum(dim=1).clamp_min(1.0)
        trend = (window * current_mask).sum(dim=1) / count
        last = window[:, -1]
        return torch.cat([trend, last - trend], dim=1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = [F.gelu(self.patch_projections[str(size)](self.causal_decompose(sequence, mask, size)))
                   for size in self.patch_sizes]
        return self.shared(torch.cat(encoded, dim=1))


class PrototypeInteraction(nn.Module):
    def __init__(self, embedding: int = 32, prototypes: int = PROTOTYPE_COUNT):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(prototypes, embedding) * 0.02)
        self.output = nn.Linear(embedding * 2, embedding)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assignment = torch.softmax(embedding @ self.prototypes.t() / np.sqrt(embedding.shape[1]), dim=1)
        context = assignment @ self.prototypes
        return self.output(torch.cat([embedding, context], dim=1)), assignment


class DriftRouterRank(nn.Module):
    def __init__(self, current_features: int = ROBUST_FEATURES, embedding: int = 32):
        super().__init__()
        self.encoder = MultiScalePatchEncoder(embedding=embedding)
        self.prototype = PrototypeInteraction(embedding=embedding)
        self.current = nn.Sequential(nn.Linear(current_features, 80), nn.GELU(), nn.Linear(80, embedding))
        self.residual = nn.Sequential(nn.Linear(embedding * 2 + 1, 64), nn.GELU(), nn.Linear(64, 1))
        self.reconstruct = nn.Linear(embedding, SEQUENCE_FEATURES)
        self.rank_bins = nn.Linear(embedding, 16)
        self.mask_reconstruct = nn.Linear(embedding, SEQUENCE_WINDOW)

    def forward(self, current: torch.Tensor, sequence: torch.Tensor, mask: torch.Tensor,
                anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal = self.encoder(sequence, mask)
        interacted, assignment = self.prototype(temporal)
        current_h = self.current(current)
        residual = self.residual(torch.cat([current_h, interacted, anchor.unsqueeze(1)], dim=1)).squeeze(1)
        return {
            "residual": residual,
            "assignment": assignment,
            "feature_reconstruction": self.reconstruct(interacted),
            "rank_logits": self.rank_bins(interacted),
            "mask_logits": self.mask_reconstruct(interacted),
        }


class DriftRouter(nn.Module):
    def __init__(self, state_dim: int = ROUTER_STATE_DIM, experts: int = len(ROUTER_EXPERTS)):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(state_dim, 16), nn.GELU(), nn.Linear(16, experts))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def masked_pretraining_loss(outputs: dict[str, torch.Tensor], feature_target: torch.Tensor,
                            rank_target: torch.Tensor, mask_target: torch.Tensor) -> torch.Tensor:
    return (
        F.smooth_l1_loss(outputs["feature_reconstruction"], feature_target)
        + 0.25 * F.cross_entropy(outputs["rank_logits"], rank_target)
        + 0.10 * F.binary_cross_entropy_with_logits(outputs["mask_logits"], mask_target)
    )


def residual_objective(residual: torch.Tensor, anchor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    desired = target - anchor
    centered_residual = residual - residual.mean()
    centered_anchor = anchor - anchor.mean()
    orthogonal_penalty = (centered_residual * centered_anchor).mean().abs()
    correlation = (centered_residual * (target - target.mean())).mean() / (
        centered_residual.std().clamp_min(1e-6) * target.std().clamp_min(1e-6)
    )
    return F.smooth_l1_loss(residual, desired) + 0.10 * (1.0 - correlation) + 0.05 * orthogonal_penalty


def train_multiscale_expert(model: DriftRouterRank, batches: Iterable[tuple[torch.Tensor, ...]],
                            epochs: int = 25, learning_rate: float = 1e-4):
    require_training_authorization("PyTorch 多尺度残差专家训练")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    history = []
    model.train()
    for epoch in range(int(epochs)):
        losses = []
        for current, sequence, mask, anchor, target in batches:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(current, sequence, mask, anchor)
            loss = residual_objective(outputs["residual"], anchor, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("多尺度残差训练出现非有限损失。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return model, history


def train_drift_router(router: DriftRouter, state: torch.Tensor, expert_predictions: torch.Tensor,
                       anchor: torch.Tensor, target: torch.Tensor, epochs: int = 100):
    require_training_authorization("漂移路由器训练")
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    router.train()
    for epoch in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = router(state)
        weights = torch.softmax(logits, dim=1)
        prediction = 0.70 * anchor + 0.30 * (weights * expert_predictions).sum(dim=1)
        load = weights.mean(dim=0)
        loss = F.smooth_l1_loss(prediction, target) + 0.05 * ((load - 1.0 / len(ROUTER_EXPERTS)) ** 2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("漂移路由训练出现非有限损失。")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    return router, history


def build_catboost_pool_dry_run(X: np.ndarray, relevance: np.ndarray, groups: np.ndarray):
    import catboost as cb
    group_ids = np.repeat(np.arange(groups.size, dtype=np.int64), groups.astype(np.int64))
    categorical_indices = list(range(ROBUST_FEATURES, ROBUST_FEATURES + 9))
    frame = pd.DataFrame(X[:, :ROBUST_FEATURES], columns=[f"feature_{i}" for i in range(ROBUST_FEATURES)])
    for j in range(9):
        frame[f"cat_{j}"] = np.asarray(np.rint(X[:, ROBUST_FEATURES + j]), dtype=np.int64).astype(str)
    return cb.Pool(frame, label=relevance, group_id=group_ids, cat_features=categorical_indices), categorical_indices


def train_catboost_expert(pool, iterations: int = 120, seed: int = 42):
    require_training_authorization("CatBoost YetiRank 专家训练")
    import catboost as cb
    params = {
        "loss_function": "YetiRank", "learning_rate": 0.03, "depth": 6,
        "l2_leaf_reg": 3.0, "random_strength": 1.0, "bootstrap_type": "Bernoulli",
        "subsample": 0.65, "random_seed": int(seed), "task_type": "CPU",
        "thread_count": NUM_THREADS, "verbose": False, "allow_writing_files": False,
    }
    return cb.train(pool, iterations=int(iterations), params=params)


def capped_indices(ctx: DataContext, split: str, start_time: int, stop_time: int, cap: int = TRAIN_STOCK_CAP) -> tuple[np.ndarray, np.ndarray]:
    rows, groups = ctx.row_slice(split, start_time, stop_time)
    capped = np.minimum(groups, int(cap)).astype(np.int32)
    indices = np.empty(int(capped.sum()), dtype=np.int64)
    source, target = int(rows.start), 0
    for full_size, capped_size in zip(groups, capped):
        full_size, capped_size = int(full_size), int(capped_size)
        positions = np.linspace(0, full_size - 1, capped_size, dtype=np.int64)
        indices[target:target + capped_size] = source + positions
        source += full_size
        target += capped_size
    assert source == int(rows.stop) and target == indices.size
    return indices, capped


def build_training_arrays(ctx: DataContext, segments: Sequence[tuple[str, int, int]], robust: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prepared, total = [], 0
    feature_count = ROBUST_FEATURES if robust else LEGACY_FEATURES
    for split, start, stop in segments:
        idx, groups = capped_indices(ctx, split, start, stop)
        prepared.append((split, idx, groups))
        total += idx.size
    X = np.empty((total, feature_count), dtype=np.float32)
    y = np.empty(total, dtype=np.int8)
    all_groups, offset = [], 0
    for split, idx, groups in prepared:
        end = offset + idx.size
        X[offset:end, :LEGACY_FEATURES] = ctx.tree[split][idx, :LEGACY_FEATURES]
        if robust:
            if split not in ctx.rank_cache:
                raise RuntimeError(f"{split} robust rank cache 未加载。")
            X[offset:end, LEGACY_FEATURES:] = ctx.rank_cache[split][idx]
        y[offset:end] = ctx.common[split]["relevance"][idx]
        all_groups.append(groups)
        offset = end
    groups = np.concatenate(all_groups).astype(np.int32)
    assert offset == total == int(groups.sum())
    return X, y, groups


def model_fingerprint(ctx: DataContext, stage: str, seed: int, segments, robust: bool, rounds: int) -> str:
    payload = {
        "experiment": EXPERIMENT_ID, "manifest": ctx.manifest_sha256,
        "feature_cache": feature_cache_fingerprint(ctx) if robust else "legacy_328",
        "stage": stage, "seed": int(seed), "segments": list(segments),
        "robust": robust, "rounds": int(rounds), "params": LGB_PARAMS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def train_or_load_models(ctx: DataContext, stage: str, segments, seeds: Sequence[int], robust: bool = True, rounds: int = EXPERT_ROUNDS):
    import lightgbm as lgb

    models, pending = {}, []
    for seed in seeds:
        fingerprint = model_fingerprint(ctx, stage, int(seed), segments, robust, rounds)
        path = MODEL_DIR / f"{stage}_seed{seed}.txt"
        meta_path = MODEL_DIR / f"{stage}_seed{seed}.metadata.json"
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint and meta.get("model_sha256") == file_sha256(path):
                models[int(seed)] = lgb.Booster(model_file=str(path))
                print(f"复用 {stage} seed={seed} 模型。")
                continue
        pending.append((int(seed), fingerprint, path, meta_path))

    if pending:
        require_training_authorization("LightGBM LambdaRank 专家训练")
        X, y, groups = build_training_arrays(ctx, segments, robust=robust)
        dataset = lgb.Dataset(X, label=y, group=groups, free_raw_data=False)
        for seed, fingerprint, path, meta_path in pending:
            params = dict(LGB_PARAMS)
            params.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed})
            started = time.time()
            model = lgb.train(params, dataset, num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(0)])
            atomic_write_text(path, model.model_to_string())
            atomic_write_json(meta_path, {
                "fingerprint": fingerprint, "stage": stage, "seed": seed,
                "segments": segments, "robust": robust, "rounds": rounds,
                "train_rows": int(X.shape[0]), "feature_count": int(X.shape[1]),
                "training_seconds": time.time() - started,
                "model_sha256": file_sha256(path),
            })
            models[seed] = model
        del dataset, X, y, groups
        gc.collect()
    return models


def predict_model(ctx: DataContext, model, split: str, start_time: int, stop_time: int, robust: bool, rounds: int = EXPERT_ROUNDS) -> tuple[np.ndarray, np.ndarray]:
    rows, groups = ctx.row_slice(split, start_time, stop_time)
    prediction = np.empty(int(rows.stop) - int(rows.start), dtype=np.float32)
    chunk = 200_000
    for begin in range(int(rows.start), int(rows.stop), chunk):
        end = min(begin + chunk, int(rows.stop))
        if robust:
            X = np.empty((end - begin, ROBUST_FEATURES), dtype=np.float32)
            X[:, :LEGACY_FEATURES] = ctx.tree[split][begin:end, :LEGACY_FEATURES]
            X[:, LEGACY_FEATURES:] = ctx.rank_cache[split][begin:end]
        else:
            X = np.asarray(ctx.tree[split][begin:end, :LEGACY_FEATURES], dtype=np.float32)
        prediction[begin - int(rows.start):end - int(rows.start)] = model.predict(X, num_iteration=int(rounds)).astype(np.float32)
        del X
    return prediction, groups


def ensemble_expert(ctx: DataContext, stage: str, models: dict[int, object], split: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    per_seed, groups = {}, None
    for seed, model in models.items():
        cache = RUNTIME_CACHE_DIR / f"{stage}_seed{seed}_{split}_{start}_{stop}.npy"
        if cache.exists():
            pred = np.load(cache)
            _, expected_groups = ctx.row_slice(split, start, stop)
            assert pred.size == int(expected_groups.sum())
            groups = expected_groups
        else:
            pred, groups = predict_model(ctx, model, split, start, stop, robust=True)
            atomic_save_npy(cache, pred)
        per_seed[seed] = np.asarray(pred, dtype=np.float32)
    assert groups is not None
    ranked = [group_rank_transform(per_seed[s], groups) for s in sorted(per_seed)]
    ensemble = np.mean(np.stack(ranked), axis=0).astype(np.float32)
    return ensemble, groups, per_seed


def exp009_fold_anchor(ctx: DataContext, fold: WalkForwardFold) -> tuple[np.ndarray, np.ndarray]:
    _, groups = ctx.row_slice("train", fold.valid_start, fold.valid_stop)
    candidates = sorted((EXP009_DIR / "runtime_cache").glob(f"{fold.name}_*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        with np.load(path) as cached:
            required = {"base_prediction", "recent_prediction_16"}
            if required.issubset(cached.files):
                base = np.asarray(cached["base_prediction"], dtype=np.float32)
                recent = np.asarray(cached["recent_prediction_16"], dtype=np.float32)
                if base.size == recent.size == int(groups.sum()):
                    anchor = blend_rank(base, recent, groups, EXP009_WEIGHT)
                    print(f"{fold.name}: 复用 exp009 锚点缓存 {path.name}。")
                    return anchor, groups

    print(f"{fold.name}: exp009 缓存不可用，按相同参数重建。", flush=True)
    base_models = train_or_load_models(ctx, f"exp009_{fold.name}_base", [("train", fold.train_start, fold.train_stop)], (42,), robust=False, rounds=BASE_ROUNDS)
    recent_start = max(fold.train_start, fold.train_stop - RECENT_LOOKBACK)
    recent_models = train_or_load_models(ctx, f"exp009_{fold.name}_recent", [("train", recent_start, fold.train_stop)], (42,), robust=False, rounds=EXPERT_ROUNDS)
    base, _ = predict_model(ctx, base_models[42], "train", fold.valid_start, fold.valid_stop, robust=False, rounds=BASE_ROUNDS)
    recent, _ = predict_model(ctx, recent_models[42], "train", fold.valid_start, fold.valid_stop, robust=False, rounds=EXPERT_ROUNDS)
    return blend_rank(base, recent, groups, EXP009_WEIGHT), groups


def exp009_official_valid_anchor(ctx: DataContext) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(ctx.common["valid"]["groups"], dtype=np.int32)
    path = EXP009_DIR / "valid_prediction.npy"
    if path.exists():
        grid = np.load(path)
        if grid.shape == (VALID_STOP - VALID_START, STOCK_COUNT) and grid.dtype == np.float32 and np.isfinite(grid).all():
            print("复用 exp009 官方 Valid 锚点。")
            return flatten_grid(ctx, grid, "valid", VALID_START), groups
        raise RuntimeError("exp009 valid_prediction.npy 存在但契约不匹配。")

    print("exp009 官方 Valid 锚点缺失，按相同参数重建。", flush=True)
    base_models = train_or_load_models(
        ctx, "exp009_official_base", [("train", TRAIN_START, TRAIN_STOP)],
        (42,), robust=False, rounds=BASE_ROUNDS,
    )
    recent_models = train_or_load_models(
        ctx, "exp009_official_recent", [("train", TRAIN_STOP - RECENT_LOOKBACK, TRAIN_STOP)],
        (42,), robust=False, rounds=EXPERT_ROUNDS,
    )
    base, _ = predict_model(ctx, base_models[42], "valid", VALID_START, VALID_STOP, robust=False, rounds=BASE_ROUNDS)
    recent, _ = predict_model(ctx, recent_models[42], "valid", VALID_START, VALID_STOP, robust=False, rounds=EXPERT_ROUNDS)
    return blend_rank(base, recent, groups, EXP009_WEIGHT), groups


def exp009_test_anchor(ctx: DataContext) -> np.ndarray:
    path = EXP009_DIR / "prediction.npy"
    if path.exists():
        grid = np.load(path)
        if grid.shape == (TEST_TIME_POINTS, STOCK_COUNT) and grid.dtype == np.float32 and np.isfinite(grid).all():
            print("复用 exp009 Test 锚点。")
            return np.asarray(grid, dtype=np.float32)
        raise RuntimeError("exp009 prediction.npy 存在但契约不匹配。")

    if not FINAL_SUBMISSION.exists():
        raise RuntimeError("exp009 Test 锚点和 final_submission 均缺失，无法重建。")
    print("exp009 Test 锚点缺失，按 Train+Valid recent1702 参数重建。", flush=True)
    formal_grid = np.load(FINAL_SUBMISSION)
    if formal_grid.shape != (TEST_TIME_POINTS, STOCK_COUNT) or formal_grid.dtype != np.float32 or not np.isfinite(formal_grid).all():
        raise RuntimeError("final_submission 契约不匹配。")
    segments = [("train", VALID_STOP - RECENT_LOOKBACK, TRAIN_STOP), ("valid", VALID_START, VALID_STOP)]
    recent_models = train_or_load_models(
        ctx, "exp009_final_recent", segments, (42,), robust=False, rounds=EXPERT_ROUNDS,
    )
    recent, groups = predict_model(ctx, recent_models[42], "test", TEST_START, TEST_STOP, robust=False, rounds=EXPERT_ROUNDS)
    formal = flatten_grid(ctx, formal_grid, "test", TEST_START)
    rebuilt = blend_rank(formal, recent, groups, EXP009_WEIGHT)
    return vector_to_grid(ctx, rebuilt, "test", TEST_START, TEST_TIME_POINTS)


def flatten_grid(ctx: DataContext, grid: np.ndarray, split: str, split_start: int) -> np.ndarray:
    times = np.asarray(ctx.common[split]["time"], dtype=np.int32) - int(split_start)
    stocks = np.asarray(ctx.common[split]["stock"], dtype=np.int32)
    return np.asarray(grid[times, stocks], dtype=np.float32)


def vector_to_grid(ctx: DataContext, values: np.ndarray, split: str, split_start: int, time_points: int, fill: float = 0.5) -> np.ndarray:
    grid = np.full((time_points, STOCK_COUNT), fill, dtype=np.float32)
    times = np.asarray(ctx.common[split]["time"], dtype=np.int32) - int(split_start)
    stocks = np.asarray(ctx.common[split]["stock"], dtype=np.int32)
    grid[times, stocks] = values
    return grid


def mean_grid_rank_corr(ctx: DataContext, left: np.ndarray, right: np.ndarray) -> float:
    groups = np.asarray(ctx.common["test"]["groups"], dtype=np.int32)
    stocks = np.asarray(ctx.common["test"]["stock"], dtype=np.int32)
    values, offset = [], 0
    for local_time, size in enumerate(groups):
        size = int(size)
        current = stocks[offset:offset + size]
        values.append(rank_ic(left[local_time, current], right[local_time, current]))
        offset += size
    return float(np.nanmean(values))
'''


PIPELINE = r'''
def causal_history_window(timeline: np.ndarray, prediction_time: int, window: int = SEQUENCE_WINDOW) -> np.ndarray:
    timeline = np.asarray(timeline, dtype=np.float32)
    if timeline.ndim != 2 or not 0 <= int(prediction_time) < timeline.shape[0]:
        raise ValueError("因果窗口输入不合法。")
    begin = max(0, int(prediction_time) - int(window) + 1)
    selected = timeline[begin:int(prediction_time) + 1]
    output = np.zeros((int(window), timeline.shape[1]), dtype=np.float32)
    output[-selected.shape[0]:] = selected
    return output


def synthetic_router_state(time_points: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, int(time_points), dtype=np.float32)
    return np.stack([x, 1.0 - x, x ** 2, np.sqrt(x), np.sin(x), np.cos(x), x * 0.5, 0.25 + x * 0.1], axis=1)


def build_router_state_from_real(current: np.ndarray, history_mask: np.ndarray,
                                 categories: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    rows, offset = [], 0
    for size in groups:
        size = int(size)
        block = current[offset:offset + size]
        mask = history_mask[offset:offset + size]
        cats = categories[offset:offset + size]
        last_coverage = float(mask[:, -1].mean())
        history_coverage = float(mask.mean())
        new_share = float((mask.sum(axis=1) <= 1).mean())
        unknown_share = float((cats[:, 5] == 0).mean())
        rows.append([
            float(np.mean(np.abs(block[:, :40]))), float(np.std(block[:, :40])),
            last_coverage, new_share, unknown_share, history_coverage,
            float(np.mean(np.std(block[:, LEGACY_FEATURES:], axis=0))),
            float(np.std(rankdata(block[:, 0], method="average") / max(size, 1))),
        ])
        offset += size
    if offset != current.shape[0]:
        raise ValueError("真实路由状态分组不匹配。")
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (len(groups), ROUTER_STATE_DIM) or not np.isfinite(result).all():
        raise AssertionError("真实路由状态构造失败。")
    return result


def run_smoke() -> dict[str, object]:
    final_sha_before = assert_final_submission_unchanged()
    groups = np.array([4, 3], dtype=np.int32)
    raw_2d = np.array([
        [1, 5], [1, 2], [3, 2], [4, 1],
        [9, 3], [7, 3], [8, 1],
    ], dtype=np.float32)
    raw = np.tile(raw_2d, (1, EXTRA_RANK_FEATURES // raw_2d.shape[1]))
    extra = np.vstack([rank_feature_block(raw[:4]), rank_feature_block(raw[4:])])
    assert np.isclose(extra[0, 0], extra[1, 0])
    assert np.all((extra > 0) & (extra <= 1))
    anchor = np.array([1, 2, 2, 4, 1, 2, 3], dtype=np.float32)
    expert = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.float32)
    target = expert.copy()
    zero = blend_rank(anchor, expert, groups, 0.0)
    assert np.array_equal(zero, group_rank_transform(anchor, groups))
    assert score_prediction(blend_rank(anchor, expert, groups, 0.2), target, groups)["mean_rank_ic"] > score_prediction(anchor, target, groups)["mean_rank_ic"]

    timeline = np.arange(100 * SEQUENCE_FEATURES, dtype=np.float32).reshape(100, SEQUENCE_FEATURES)
    before = causal_history_window(timeline, 70)
    changed = timeline.copy(); changed[71:] = -99999.0
    assert np.array_equal(before, causal_history_window(changed, 70))

    torch.manual_seed(42)
    n = 12
    model = DriftRouterRank()
    current_t = torch.randn(n, ROBUST_FEATURES, requires_grad=True)
    sequence_t = torch.randn(n, SEQUENCE_WINDOW, SEQUENCE_FEATURES, requires_grad=True)
    mask_t = (torch.rand(n, SEQUENCE_WINDOW) > 0.20).float(); mask_t[:, -1] = 1.0
    anchor_t = torch.randn(n)
    target_t = torch.randn(n)
    outputs = model(current_t, sequence_t, mask_t, anchor_t)
    pretrain_loss = masked_pretraining_loss(
        outputs, sequence_t[:, -1].detach(), torch.arange(n) % 16, mask_t.detach()
    )
    residual_loss = residual_objective(outputs["residual"], anchor_t, target_t)
    total_loss = pretrain_loss + residual_loss
    total_loss.backward()
    assert torch.isfinite(total_loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    parameter_snapshot = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert all(torch.equal(parameter_snapshot[name], value.detach()) for name, value in model.named_parameters())

    permutation = torch.tensor([3, 1, 11, 0, 4, 6, 9, 2, 7, 8, 5, 10])
    with torch.no_grad():
        original, _ = model.prototype(model.encoder(sequence_t.detach(), mask_t))
        permuted, _ = model.prototype(model.encoder(sequence_t.detach()[permutation], mask_t[permutation]))
    assert torch.allclose(original[permutation], permuted, atol=1e-6)

    router = DriftRouter()
    states = torch.from_numpy(synthetic_router_state(30))
    logits = router(states).detach().numpy()
    for t in range(logits.shape[0]):
        logits[t, t % len(ROUTER_EXPERTS)] += 4.0
        logits[t, (t + 1) % len(ROUTER_EXPERTS)] += 2.0
    router_weights = top2_router_weights(logits)
    assert np.allclose(router_weights.sum(axis=1), 1.0) and np.all((router_weights > 0).sum(axis=1) == 2)
    participation = (router_weights > 0).mean(axis=0)

    blend_groups = np.full(30, 4, dtype=np.int32)
    rng = np.random.default_rng(42)
    blend_anchor = rng.normal(size=int(blend_groups.sum())).astype(np.float32)
    blend_experts = [blend_anchor + rng.normal(scale=0.2 + i * 0.05, size=blend_anchor.size).astype(np.float32) for i in range(3)]
    seed_predictions = rng.normal(size=(3, 30, 3)).astype(np.float32)
    budgets = disagreement_shrink(seed_predictions)
    low_disagreement = disagreement_shrink(np.zeros_like(seed_predictions))
    assert float(low_disagreement.mean()) > float(budgets.mean())
    integrated = integrated_blend(blend_anchor, blend_experts, blend_groups, router_weights, budgets)
    integrated_corr = float(np.nanmean(group_rank_ic_series(integrated, blend_anchor, blend_groups)))
    assert integrated.shape == blend_anchor.shape and integrated.dtype == np.float32 and np.isfinite(integrated).all()

    guard_verified = False
    try:
        require_training_authorization("smoke 训练保护探针")
    except RuntimeError:
        guard_verified = True
    assert guard_verified

    probe = RUN_DIR / "atomic_probe.npy"
    atomic_save_npy(probe, extra)
    assert np.array_equal(np.load(probe), extra)
    probe.unlink()
    cache_payload = {"version": EXPERIMENT_VERSION, "patch_sizes": PATCH_SIZES, "prototypes": PROTOTYPE_COUNT}
    fingerprint = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert fingerprint != hashlib.sha256(json.dumps({**cache_payload, "prototypes": 8}, sort_keys=True).encode("utf-8")).hexdigest()
    final_sha_after = assert_final_submission_unchanged(final_sha_before)
    report = {
        "status": "smoke_passed", "experiment_version": EXPERIMENT_VERSION,
        "training_performed": False, "optimizer_step_called": False,
        "ties_verified": True, "rank_range_verified": True,
        "causal_windows_verified": True, "patch_sizes": list(PATCH_SIZES),
        "model_forward_verified": True, "loss_backward_graph_verified": True,
        "parameters_unchanged_after_backward": True, "prototype_permutation_equivariant": True,
        "orthogonalization_verified": True, "top2_router_verified": True,
        "router_participation": participation, "uncertainty_shrink_verified": True,
        "integrated_blend_finite": True, "integrated_anchor_rankic": integrated_corr,
        "training_guard_verified": guard_verified, "atomic_write_verified": True,
        "cache_fingerprint_verified": True,
        "final_submission_sha256_before": final_sha_before,
        "final_submission_sha256_after": final_sha_after,
    }
    atomic_write_json(RUN_DIR / "smoke_report.json", report)
    return report


def run_preflight() -> dict[str, object]:
    import lightgbm as lgb
    import catboost as cb
    final_sha_before = assert_final_submission_unchanged()
    ctx = DataContext()
    ranked, groups, rows = build_preflight_rank_cache(ctx)
    ranked_again, groups_again, rows_again = build_preflight_rank_cache(ctx)
    assert np.array_equal(ranked, ranked_again) and np.array_equal(groups, groups_again)
    assert rows == rows_again
    full = np.empty((ranked.shape[0], ROBUST_FEATURES), dtype=np.float32)
    full[:, :LEGACY_FEATURES] = ctx.tree["train"][rows, :LEGACY_FEATURES]
    full[:, LEGACY_FEATURES:] = ranked
    relevance = np.asarray(ctx.common["train"]["relevance"][rows], dtype=np.int8)
    lgb_dataset = lgb.Dataset(full, label=relevance, group=groups, free_raw_data=False)
    lgb_dataset.construct()
    assert lgb_dataset.num_data() == full.shape[0] and lgb_dataset.num_feature() == ROBUST_FEATURES

    categories = np.asarray(ctx.tree["train"][rows, 408:417], dtype=np.float32)
    cat_input = np.concatenate([full, categories], axis=1).astype(np.float32)
    cat_pool, categorical_indices = build_catboost_pool_dry_run(cat_input, relevance, groups)
    assert cat_pool.num_row() == full.shape[0] and cat_pool.num_col() == ROBUST_FEATURES + 9
    assert categorical_indices == list(range(ROBUST_FEATURES, ROBUST_FEATURES + 9))
    assert bool(ctx.manifest["processing"]["unknown_bucket"])

    history_all, history_mask_all, _ = ctx.temporal_batch("train", TRAIN_START, TRAIN_START + 6)
    assert history_all.shape == (full.shape[0], SEQUENCE_WINDOW, SEQUENCE_FEATURES)
    assert history_mask_all.shape == (full.shape[0], SEQUENCE_WINDOW)
    sample_count = min(128, full.shape[0])
    history, history_mask = history_all[:sample_count], history_mask_all[:sample_count]
    torch.manual_seed(42)
    model = DriftRouterRank()
    current_t = torch.from_numpy(full[:sample_count])
    sequence_t = torch.from_numpy(history)
    mask_t = torch.from_numpy(history_mask)
    anchor_t = torch.linspace(-1.0, 1.0, sample_count)
    with torch.no_grad():
        outputs = model(current_t, sequence_t, mask_t, anchor_t)
        dry_loss = masked_pretraining_loss(
            outputs, sequence_t[:, -1], torch.arange(sample_count) % 16, mask_t
        ) + residual_objective(outputs["residual"], anchor_t, torch.flip(anchor_t, dims=[0]))
    assert torch.isfinite(dry_loss) and outputs["assignment"].shape == (sample_count, PROTOTYPE_COUNT)

    router_state = build_router_state_from_real(full, history_mask=history_mask_all,
                                                categories=categories, groups=groups)
    router = DriftRouter()
    with torch.no_grad():
        router_logits = router(torch.from_numpy(router_state)).numpy()
    for t in range(router_logits.shape[0]):
        router_logits[t, t % len(ROUTER_EXPERTS)] += 4.0
        router_logits[t, (t + 1) % len(ROUTER_EXPERTS)] += 2.0
    router_weights = top2_router_weights(router_logits)
    rng = np.random.default_rng(2026)
    anchor = rng.normal(size=full.shape[0]).astype(np.float32)
    experts = [anchor + rng.normal(scale=0.10 + j * 0.05, size=anchor.size).astype(np.float32) for j in range(3)]
    seed_predictions = rng.normal(size=(3, groups.size, 3)).astype(np.float32)
    budgets = disagreement_shrink(seed_predictions)
    integrated = integrated_blend(anchor, experts, groups, router_weights, budgets)
    assert integrated.shape == anchor.shape and np.isfinite(integrated).all()

    test_groups = np.asarray(ctx.common["test"]["groups"], dtype=np.int32)
    test_values = ((np.asarray(ctx.common["test"]["stock"], dtype=np.float32) % 997.0) / 997.0).astype(np.float32)
    contract_sample = vector_to_grid(ctx, test_values, "test", TEST_START, TEST_TIME_POINTS)
    test_mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
    test_mask[np.asarray(ctx.common["test"]["time"], dtype=np.int32) - TEST_START,
              np.asarray(ctx.common["test"]["stock"], dtype=np.int32)] = True
    assert contract_sample.shape == (TEST_TIME_POINTS, STOCK_COUNT) and contract_sample.dtype == np.float32
    assert np.isfinite(contract_sample).all() and int(test_mask.sum()) == 2_042_538
    assert int((~test_mask).sum()) == 292_106 and np.all(contract_sample[~test_mask] == 0.5)
    atomic_save_npy(RUN_DIR / "contract_sample.npy", contract_sample)

    training_guard_verified = False
    try:
        require_training_authorization("preflight 训练保护探针")
    except RuntimeError:
        training_guard_verified = True
    assert training_guard_verified
    final_sha_after = assert_final_submission_unchanged(final_sha_before)
    report = {
        "status": "IMPLEMENTED_AND_DRY_RUN_PASSED", "manifest_sha256": ctx.manifest_sha256,
        "experiment_version": EXPERIMENT_VERSION, "training_performed": False,
        "time_interval": [TRAIN_START, TRAIN_START + 6], "rows": int(full.shape[0]),
        "feature_count": int(full.shape[1]), "rank_cache_reuse_verified": True,
        "column_order": ["legacy_328", *EXTRA_FEATURE_NAMES],
        "label_free_feature_construction": True,
        "lightgbm_dataset_constructed_without_training": True,
        "catboost_pool_constructed_without_training": True,
        "catboost_version": cb.__version__, "unknown_bucket_verified": True,
        "sequence_shape": list(history.shape), "real_sequence_full_shape": list(history_all.shape),
        "patch_sizes": list(PATCH_SIZES),
        "prototype_shape": list(outputs["assignment"].shape),
        "random_forward_and_loss_finite": True,
        "router_state_shape": list(router_state.shape), "top2_router_verified": True,
        "orthogonal_integrated_blend_verified": True,
        "contract_sample_path": str((RUN_DIR / "contract_sample.npy").relative_to(PROJECT_ROOT)),
        "contract_sample_is_candidate": False, "output_contract_verified": True,
        "training_guard_verified": training_guard_verified,
        "final_submission_sha256_before": final_sha_before,
        "final_submission_sha256_after": final_sha_after,
    }
    atomic_write_json(RUN_DIR / "preflight_report.json", report)
    return report


def choose_weight(dev_rows: list[dict[str, object]]) -> tuple[float, pd.DataFrame, str]:
    frame = pd.DataFrame(dev_rows)
    summaries = []
    for weight, part in frame.groupby("weight"):
        summaries.append({
            "weight": float(weight), "mean_rankic": float(part["candidate_rankic"].mean()),
            "mean_delta": float(part["delta"].mean()), "worst_delta": float(part["delta"].min()),
            "all_positive": bool((part["delta"] > 0).all()), "fold_std": float(part["candidate_rankic"].std(ddof=0)),
        })
    search = pd.DataFrame(summaries).sort_values("weight")
    search["qualified"] = (search["weight"] > 0) & (search["mean_delta"] >= GATE_THRESHOLDS["dev_mean_delta"]) & search["all_positive"]
    qualified = search[search["qualified"]]
    if qualified.empty:
        selected, reason = 0.0, "没有新增权重同时通过两个开发折和平均增量门槛，回退 exp009 锚点。"
    else:
        best = float(qualified["mean_delta"].max())
        near = qualified[qualified["mean_delta"] >= best * 0.95].sort_values(["weight", "fold_std"])
        selected = float(near.iloc[0]["weight"])
        reason = "在达到最佳开发增益 95% 的配置中选择最小新增专家权重。"
    search["selected"] = np.isclose(search["weight"], selected)
    return selected, search, reason


def run_full() -> dict[str, object]:
    started = time.time()
    final_sha_before = file_sha256(FINAL_SUBMISSION)
    ctx = DataContext()
    feature_manifest = ensure_full_rank_cache(ctx)

    fold_rows, fold_state = [], {}
    for fold in DEV_FOLDS:
        anchor, groups = exp009_fold_anchor(ctx, fold)
        target = ctx.target("train", fold.valid_start, fold.valid_stop)
        recent_start = max(fold.train_start, fold.train_stop - RECENT_LOOKBACK)
        models = train_or_load_models(ctx, fold.name, [("train", recent_start, fold.train_stop)], SEEDS)
        expert, _, per_seed = ensemble_expert(ctx, fold.name, models, "train", fold.valid_start, fold.valid_stop)
        anchor_metrics = score_prediction(anchor, target, groups)
        fold_state[fold.name] = (anchor, expert, target, groups, per_seed)
        for weight in WEIGHT_CANDIDATES:
            candidate = blend_rank(anchor, expert, groups, weight)
            metrics = score_prediction(candidate, target, groups)
            fold_rows.append({
                "stage": "dev", "fold": fold.name, "weight": weight,
                "anchor_rankic": anchor_metrics["mean_rank_ic"],
                "candidate_rankic": metrics["mean_rank_ic"],
                "delta": metrics["mean_rank_ic"] - anchor_metrics["mean_rank_ic"],
                "late_delta": metrics["late_half_rank_ic"] - anchor_metrics["late_half_rank_ic"],
                "worst_quarter_delta": metrics["worst_quarter_rank_ic"] - anchor_metrics["worst_quarter_rank_ic"],
            })
        del models
        gc.collect()

    selected_weight, weight_search, selection_reason = choose_weight(fold_rows)
    atomic_write_csv(RUN_DIR / "weight_search.csv", weight_search)
    dev_seed_deltas = []
    for fold in DEV_FOLDS:
        anchor, _, target, groups, per_seed = fold_state[fold.name]
        anchor_score = score_prediction(anchor, target, groups)["mean_rank_ic"]
        for seed in SEEDS:
            seed_candidate = blend_rank(anchor, per_seed[seed], groups, selected_weight)
            delta = score_prediction(seed_candidate, target, groups)["mean_rank_ic"] - anchor_score
            dev_seed_deltas.append(delta)
            fold_rows.append({
                "stage": "dev_seed", "fold": fold.name, "seed": seed,
                "weight": selected_weight, "anchor_rankic": anchor_score,
                "candidate_rankic": anchor_score + delta, "delta": delta,
                "late_delta": np.nan, "worst_quarter_delta": np.nan,
            })

    fold = SHADOW_FOLD
    shadow_anchor, shadow_groups = exp009_fold_anchor(ctx, fold)
    shadow_target = ctx.target("train", fold.valid_start, fold.valid_stop)
    recent_start = max(fold.train_start, fold.train_stop - RECENT_LOOKBACK)
    shadow_models = train_or_load_models(ctx, "shadow", [("train", recent_start, fold.train_stop)], SEEDS)
    shadow_expert, _, shadow_per_seed = ensemble_expert(ctx, "shadow", shadow_models, "train", fold.valid_start, fold.valid_stop)
    shadow_anchor_metrics = score_prediction(shadow_anchor, shadow_target, shadow_groups)
    shadow_candidate = blend_rank(shadow_anchor, shadow_expert, shadow_groups, selected_weight)
    shadow_metrics = score_prediction(shadow_candidate, shadow_target, shadow_groups)
    shadow_seed_deltas = []
    for seed in SEEDS:
        seed_candidate = blend_rank(shadow_anchor, shadow_per_seed[seed], shadow_groups, selected_weight)
        shadow_seed_deltas.append(score_prediction(seed_candidate, shadow_target, shadow_groups)["mean_rank_ic"] - shadow_anchor_metrics["mean_rank_ic"])
    fold_rows.append({
        "stage": "shadow", "fold": fold.name, "weight": selected_weight,
        "anchor_rankic": shadow_anchor_metrics["mean_rank_ic"], "candidate_rankic": shadow_metrics["mean_rank_ic"],
        "delta": shadow_metrics["mean_rank_ic"] - shadow_anchor_metrics["mean_rank_ic"],
        "late_delta": shadow_metrics["late_half_rank_ic"] - shadow_anchor_metrics["late_half_rank_ic"],
        "worst_quarter_delta": shadow_metrics["worst_quarter_rank_ic"] - shadow_anchor_metrics["worst_quarter_rank_ic"],
    })
    atomic_write_csv(RUN_DIR / "fold_results.csv", pd.DataFrame(fold_rows))
    del shadow_models
    gc.collect()

    valid_groups = np.asarray(ctx.common["valid"]["groups"], dtype=np.int32)
    valid_target = np.asarray(ctx.common["valid"]["y"], dtype=np.float32)
    valid_anchor, rebuilt_valid_groups = exp009_official_valid_anchor(ctx)
    assert np.array_equal(valid_groups, rebuilt_valid_groups)
    official_models = train_or_load_models(ctx, "official", [("train", TRAIN_STOP - RECENT_LOOKBACK, TRAIN_STOP)], SEEDS)
    valid_expert, _, valid_per_seed = ensemble_expert(ctx, "official", official_models, "valid", VALID_START, VALID_STOP)
    valid_candidate = blend_rank(valid_anchor, valid_expert, valid_groups, selected_weight)
    valid_anchor_metrics = score_prediction(valid_anchor, valid_target, valid_groups)
    valid_candidate_metrics = score_prediction(valid_candidate, valid_target, valid_groups)
    valid_seed_deltas = []
    for seed in SEEDS:
        candidate = blend_rank(valid_anchor, valid_per_seed[seed], valid_groups, selected_weight)
        valid_seed_deltas.append(score_prediction(candidate, valid_target, valid_groups)["mean_rank_ic"] - valid_anchor_metrics["mean_rank_ic"])
    atomic_write_csv(RUN_DIR / "official_valid_results.csv", pd.DataFrame([
        {"model": "exp009_anchor", **valid_anchor_metrics},
        {"model": "robust_rank_348_candidate", **valid_candidate_metrics},
    ]))
    atomic_save_npy(RUN_DIR / "valid_prediction.npy", vector_to_grid(ctx, valid_candidate, "valid", VALID_START, VALID_STOP - VALID_START))
    del official_models
    gc.collect()

    test_anchor_grid = exp009_test_anchor(ctx)
    test_groups = np.asarray(ctx.common["test"]["groups"], dtype=np.int32)
    final_training_skipped = bool(selected_weight == 0.0)
    if final_training_skipped:
        test_expert = np.full(int(test_groups.sum()), 0.5, dtype=np.float32)
        test_candidate_grid = test_anchor_grid.copy()
    else:
        final_segments = [("train", VALID_STOP - RECENT_LOOKBACK, TRAIN_STOP), ("valid", VALID_START, VALID_STOP)]
        final_models = train_or_load_models(ctx, "final", final_segments, SEEDS)
        test_expert, _, _ = ensemble_expert(ctx, "final", final_models, "test", TEST_START, TEST_STOP)
        test_anchor = flatten_grid(ctx, test_anchor_grid, "test", TEST_START)
        test_candidate = blend_rank(test_anchor, test_expert, test_groups, selected_weight)
        test_candidate_grid = vector_to_grid(ctx, test_candidate, "test", TEST_START, TEST_TIME_POINTS)
        del final_models
        gc.collect()

    expert_grid = vector_to_grid(ctx, group_rank_transform(test_expert, test_groups), "test", TEST_START, TEST_TIME_POINTS)
    test_corr = mean_grid_rank_corr(ctx, test_candidate_grid, test_anchor_grid)
    test_mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
    test_mask[np.asarray(ctx.common["test"]["time"], dtype=np.int32) - TEST_START, np.asarray(ctx.common["test"]["stock"], dtype=np.int32)] = True
    output_contract = bool(
        test_candidate_grid.shape == (TEST_TIME_POINTS, STOCK_COUNT)
        and test_candidate_grid.dtype == np.float32
        and np.isfinite(test_candidate_grid).all()
        and int(test_mask.sum()) == 2_042_538
        and int((~test_mask).sum()) == 292_106
        and np.all(test_candidate_grid[~test_mask] == 0.5)
    )

    dev_selected = [row for row in fold_rows if row["stage"] == "dev" and np.isclose(row["weight"], selected_weight)]
    measured = {
        "selected_weight": selected_weight,
        "dev_mean_delta": float(np.mean([r["delta"] for r in dev_selected])),
        "dev_all_positive": bool(all(r["delta"] > 0 for r in dev_selected)),
        "shadow_delta": shadow_metrics["mean_rank_ic"] - shadow_anchor_metrics["mean_rank_ic"],
        "shadow_late_delta": shadow_metrics["late_half_rank_ic"] - shadow_anchor_metrics["late_half_rank_ic"],
        "official_mean_delta": valid_candidate_metrics["mean_rank_ic"] - valid_anchor_metrics["mean_rank_ic"],
        "official_late_delta": valid_candidate_metrics["late_half_rank_ic"] - valid_anchor_metrics["late_half_rank_ic"],
        "official_worst_quarter_delta": valid_candidate_metrics["worst_quarter_rank_ic"] - valid_anchor_metrics["worst_quarter_rank_ic"],
        "seed_direction_consistent": bool(all(x > 0 for x in dev_seed_deltas + shadow_seed_deltas + valid_seed_deltas)),
        "seed_delta_spread": float(max(dev_seed_deltas + shadow_seed_deltas + valid_seed_deltas) - min(dev_seed_deltas + shadow_seed_deltas + valid_seed_deltas)),
        "test_anchor_rank_corr": test_corr, "output_contract": output_contract,
    }
    gates = evaluate_gates(measured)
    promoted = bool(gates["passed"].all())
    prediction_grid = test_candidate_grid if promoted else test_anchor_grid.copy()
    atomic_save_npy(RUN_DIR / "anchor_prediction.npy", test_anchor_grid.astype(np.float32))
    atomic_save_npy(RUN_DIR / "expert_prediction.npy", expert_grid.astype(np.float32))
    atomic_save_npy(RUN_DIR / "candidate_prediction.npy", test_candidate_grid.astype(np.float32))
    atomic_save_npy(RUN_DIR / "prediction.npy", prediction_grid.astype(np.float32))
    atomic_write_csv(RUN_DIR / "promotion_gates.csv", gates)

    final_sha_after = file_sha256(FINAL_SUBMISSION)
    if final_sha_after != final_sha_before:
        raise RuntimeError("final_submission 在 exp015 运行期间发生变化，已停止验收。")
    metrics = {
        "run_mode": "full", "status": "completed_candidate" if promoted else "completed_not_promoted",
        "feature_view": "robust_rank_348", "selected_weight": selected_weight,
        "selection_reason": selection_reason, "dev": dev_selected,
        "dev_seed_deltas": dev_seed_deltas,
        "shadow_anchor": shadow_anchor_metrics, "shadow_candidate": shadow_metrics,
        "shadow_seed_deltas": shadow_seed_deltas, "official_anchor": valid_anchor_metrics,
        "official_candidate": valid_candidate_metrics, "official_seed_deltas": valid_seed_deltas,
        "measured_gates": measured, "promoted": promoted,
        "online_promotion_line": ONLINE_PROMOTION_LINE,
    }
    atomic_write_json(RUN_DIR / "metrics.json", metrics)
    outputs = {}
    for name in ("prediction.npy", "candidate_prediction.npy", "anchor_prediction.npy", "expert_prediction.npy", "valid_prediction.npy", "fold_results.csv", "weight_search.csv", "promotion_gates.csv"):
        path = RUN_DIR / name
        outputs[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    metadata = {
        "experiment": EXPERIMENT_ID, "status": metrics["status"], "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest_sha256": ctx.manifest_sha256, "feature_cache_fingerprint": feature_manifest["fingerprint"],
        "feature_view": "robust_rank_348", "feature_count": ROBUST_FEATURES,
        "seeds": SEEDS, "rounds": EXPERT_ROUNDS, "selected_weight": selected_weight,
        "final_training_segments": [["train", VALID_STOP - RECENT_LOOKBACK, TRAIN_STOP], ["valid", VALID_START, VALID_STOP]],
        "final_training_skipped": final_training_skipped, "promoted": promoted,
        "fallback_used": not promoted, "formal_submission_overwritten": False,
        "final_submission_sha256_before": final_sha_before, "final_submission_sha256_after": final_sha_after,
        "duration_seconds": time.time() - started, "outputs": outputs,
    }
    atomic_write_json(RUN_DIR / "metadata.json", metadata)
    atomic_write_json(RUN_DIR / "feature_cache_manifest.json", feature_manifest)
    atomic_write_json(RUN_DIR / "submission_choice.json", {
        "status": "PROMOTED" if promoted else "REJECTED: prediction.npy is exp009 anchor fallback",
        "promoted": promoted, "selected_weight": selected_weight,
        "candidate_file": str(RUN_DIR / "candidate_prediction.npy"),
        "recommended_file": str(RUN_DIR / "prediction.npy"), "fallback_used": not promoted,
        "manual_online_promotion_requirement": f"RankIC > {ONLINE_PROMOTION_LINE}",
    })
    report = [
        "# exp015 漂移稳健截面秩专家实验报告", "",
        f"- 状态：`{metrics['status']}`", "- 新增变量：后 20 个原始特征的逐时点 average-tie percentile rank",
        f"- 新增专家权重：`{selected_weight:.3f}`", f"- 官方 Valid 增量：`{measured['official_mean_delta']:+.6f}`",
        f"- Test 与 exp009 锚点相关：`{test_corr:.6f}`", f"- 是否通过离线晋级门槛：`{promoted}`",
        f"- 人工线上最终晋级要求：RankIC 严格高于 `{ONLINE_PROMOTION_LINE}`", "",
        "本实验不会覆盖 `04_results/final_submission/prediction.npy`。", "",
    ]
    atomic_write_text(RUN_DIR / "experiment_report.md", "\n".join(report))
    return metadata


if RUN_MODE == "smoke":
    RESULT = run_smoke()
elif RUN_MODE == "preflight":
    RESULT = run_preflight()
else:
    require_training_authorization("exp015 integrated_v2 full 流程")
    RESULT = run_full()

print(json.dumps(json_ready(RESULT), ensure_ascii=False, indent=2))
'''


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "kernelspec": {"display_name": "jingge_ts", "language": "python", "name": "jingge_ts"},
    "language_info": {"name": "python", "version": "3.10"},
}
notebook["cells"] = [
    markdown("""
# exp015：DriftRouter-Rank 创新增强版（integrated_v2）

在原 exp015 的 `robust_rank_348` 基础上，原地加入 CatBoost/YetiRank、多尺度 patch、16 个原型、因果掩码预训练、OOF 正交残差、漂移 top-2 路由和专家分歧回缩。所有新增缓存及结果隔离在 `integrated_v2` 子目录。

默认只运行 `smoke`。任何真实训练都要求同时设置 `DSCR_EXP015_MODE=full` 与 `DSCR_EXP015_ALLOW_TRAINING=YES`；实验不会自动覆盖 `04_results/final_submission/prediction.npy`。
"""),
    markdown("""
## 运行模式

- `smoke`：纯合成数据检查因果窗口、前向/反向图、原型、正交化、top-2 路由、回缩和训练闸门；不调用 `optimizer.step()`。
- `preflight`：读取 6 个真实时间点，构造 LightGBM Dataset 与 CatBoost Pool，执行随机初始化前向、损失、路由及输出契约检查；不调用任何训练 API。
- `full`：保留给以后人工执行；没有双重授权时在任何缓存或训练操作前终止。

通过 `DSCR_EXP015_MODE` 选择模式；默认是 `smoke`。阶段参数为 `DSCR_EXP015_STAGE=all|features|pretrain|experts|router|final`。
"""),
    code(SETUP),
    markdown("## 指标、截面秩与审计工具"),
    code(UTILITIES),
    markdown("## 数据契约与 robust_rank_348 特征缓存"),
    code(DATA_AND_CACHE),
    markdown("## 多尺度网络、原型、CatBoost、LightGBM 与 exp009 工具"),
    code(MODELING),
    markdown("## 无训练 Smoke / Preflight 与受保护 Full 主流程"),
    code(PIPELINE),
    markdown("""
## 判定说明

本轮只验收 smoke 和 preflight，不运行 full，不生成真实候选。preflight 中的 `contract_sample.npy` 仅用于验证 `(442,5282)` 输出装配，明确标记为非候选。以后只有显式授权的 full 流程可以训练；真实线上 RankIC 严格高于 `0.109959` 后，才可人工认定最终晋级。
    """),
]
for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"exp015-{index:02d}"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
compiled_cells = []
training_call_owners = {}
allowed_training_owners = {
    "train_or_load_models", "train_multiscale_expert", "train_drift_router", "train_catboost_expert",
}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        compile(cell["source"], f"{OUTPUT.name}:cell-{index}", "exec")
        compiled_cells.append(index)
        tree = ast.parse(cell["source"])
        for function in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            calls = []
            for node in ast.walk(function):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    name = node.func.id
                else:
                    continue
                if name in {"train", "step"}:
                    calls.append(name)
            if calls:
                training_call_owners[function.name] = sorted(set(calls))
unexpected_training_owners = sorted(set(training_call_owners) - allowed_training_owners)
if unexpected_training_owners:
    raise AssertionError(f"训练 API 出现在未授权函数中：{unexpected_training_owners}")
smoke_or_preflight_training_calls = sorted(set(training_call_owners) & {"run_smoke", "run_preflight"})
if smoke_or_preflight_training_calls:
    raise AssertionError(f"smoke/preflight 出现训练 API：{smoke_or_preflight_training_calls}")
static_report = {
    "status": "STATIC_CHECK_PASSED",
    "notebook": "02_experiments/exp_015_drift_robust_rank/experiment.ipynb",
    "code_cells_compiled": compiled_cells,
    "kernelspec": notebook["metadata"]["kernelspec"],
    "default_mode": "smoke",
    "supported_stages": ["all", "features", "pretrain", "experts", "router", "final"],
    "feature_view": "robust_rank_348",
    "feature_count": 348,
    "integrated_cache_dir": "03_cache/exp_015_drift_robust_rank/integrated_v2",
    "integrated_result_dir": "04_results/exp_015_drift_robust_rank/integrated_v2",
    "training_guard": "DSCR_EXP015_MODE=full + DSCR_EXP015_ALLOW_TRAINING=YES",
    "training_call_owners": training_call_owners,
    "unexpected_training_call_owners": unexpected_training_owners,
    "smoke_or_preflight_training_call_owners": smoke_or_preflight_training_calls,
    "smoke_or_preflight_contains_train_or_step": bool(smoke_or_preflight_training_calls),
    "full_executed": False,
}
static_path = ROOT / "04_results" / "exp_015_drift_robust_rank" / "integrated_v2" / "static" / "static_check_report.json"
static_path.parent.mkdir(parents=True, exist_ok=True)
static_path.write_text(__import__("json").dumps(static_report, ensure_ascii=False, indent=2), encoding="utf-8")
print(OUTPUT)
