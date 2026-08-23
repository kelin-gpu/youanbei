from __future__ import annotations

import gc
import json
import math
import mmap
import os
import platform
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
import zstandard as zstd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge


T = 3603
S = 5282
NUMERIC_FEATURE_COUNT = 99
CATEGORY_FEATURE_COUNT = 9
TRAIN_START = 486
VALID_START = 2918
TEST_START = 3161
LINEAR_BASELINE_IC = 0.089678
TCN_BASELINE_IC = 0.091556


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: int
    train_stop: int
    valid_start: int
    valid_stop: int


FOLDS = (
    WalkForwardFold("fold_1", 486, 1702, 1702, 1945),
    WalkForwardFold("fold_2", 486, 2188, 2188, 2431),
    WalkForwardFold("fold_3", 486, 2674, 2674, 2918),
)


@dataclass
class PipelineConfig:
    data_path: str = "data.z"
    output_dir: str = "y1_rank_outputs"
    cache_dir: str = ".cache_y1_rank"
    seed: int = 42
    top_feature_count: int = 40
    history_feature_count: int = 20
    interaction_pair_count: int = 50
    interaction_keep_count: int = 30
    correlation_time_samples: int = 64
    correlation_stock_cap: int = 1200
    discovery_train_time_samples: int = 128
    discovery_stock_cap: int = 600
    ablation_stock_cap: int = 400
    tuning_stock_cap: int = 800
    final_train_stock_cap: int = 1200
    tuning_trials: int = 30
    verify_top_parameter_sets: int = 5
    max_boost_rounds: int = 3000
    early_stopping_rounds: int = 100
    imputation_ic_threshold: float = 0.0005
    interaction_ic_threshold: float = 0.001
    cleanup_feature_matrices: bool = True
    verbose: bool = True

    @property
    def data_path_obj(self) -> Path:
        return Path(self.data_path).resolve()

    @property
    def output_dir_obj(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def cache_dir_obj(self) -> Path:
        return Path(self.cache_dir).resolve()


def log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_booster(booster: lgb.Booster, destination: Path) -> None:
    """Save through an ASCII-only temporary path for LightGBM on Windows."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.gettempdir()) / "y1_rank_pipeline"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / f"model_{os.getpid()}_{time.time_ns()}.txt"
    try:
        booster.save_model(str(temporary_path))
        shutil.copyfile(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_booster(model_path: str | Path) -> lgb.Booster:
    """Load a model through an ASCII-only temporary path on Windows."""
    source = Path(model_path)
    if not source.exists():
        raise FileNotFoundError(source)
    temporary_dir = Path(tempfile.gettempdir()) / "y1_rank_pipeline"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / f"load_{os.getpid()}_{time.time_ns()}.txt"
    try:
        shutil.copyfile(source, temporary_path)
        return lgb.Booster(model_file=str(temporary_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class LimitedReader:
    def __init__(self, raw, remaining: int):
        self.raw = raw
        self.remaining = int(remaining)

    def readable(self):
        return True

    def read(self, size: int = -1):
        if self.remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self.remaining
        size = min(int(size), self.remaining)
        chunk = self.raw.read(size)
        self.remaining -= len(chunk)
        return chunk


def ensure_uncompressed_pickle(config: PipelineConfig) -> Path:
    cache_dir = config.cache_dir_obj
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / "payload.pkl"
    expected_size = 9_096_840_663
    if output_path.exists() and output_path.stat().st_size == expected_size:
        return output_path

    source_path = config.data_path_obj
    log("Streaming data.z to a memory-mappable pickle cache", config.verbose)
    with source_path.open("rb") as source:
        header = source.read(7)
        if len(header) != 7 or header[:3] != b"\x80\x05\x42":
            raise ValueError("Unexpected data.z outer pickle header")
        payload_length = struct.unpack("<I", header[3:7])[0]
        limited = LimitedReader(source, payload_length)
        decompressor = zstd.ZstdDecompressor()
        temporary_path = output_path.with_suffix(".pkl.partial")
        with decompressor.stream_reader(limited) as reader, temporary_path.open("wb") as target:
            shutil.copyfileobj(reader, target, length=16 * 1024 * 1024)
        temporary_path.replace(output_path)

    if output_path.stat().st_size != expected_size:
        raise ValueError(
            f"Unexpected payload size {output_path.stat().st_size}; expected {expected_size}"
        )
    return output_path


def _locate_raw_array(mm: mmap.mmap, key: bytes, byte_length: int) -> int:
    key_pattern = bytes([0x8C, len(key)]) + key + b"\x94"
    if byte_length <= np.iinfo(np.uint32).max:
        data_pattern = b"\x42" + struct.pack("<I", byte_length)
    else:
        data_pattern = b"\x8E" + struct.pack("<Q", byte_length)
    search_from = 0
    while True:
        key_position = mm.find(key_pattern, search_from)
        if key_position < 0:
            break
        opcode_position = mm.find(data_pattern, key_position, key_position + 512)
        if opcode_position >= 0:
            return opcode_position + len(data_pattern)
        search_from = key_position + 1
    raise ValueError(f"Could not locate raw array {key!r}")


class CompetitionData:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pickle_path = ensure_uncompressed_pickle(config)
        n = T * S
        with self.pickle_path.open("rb") as handle:
            mm = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            offsets = {
                "num_x": _locate_raw_array(mm, b"num_x", n * NUMERIC_FEATURE_COUNT * 4),
                "cat_x": _locate_raw_array(mm, b"cat_x", n * CATEGORY_FEATURE_COUNT * 8),
                "y1": _locate_raw_array(mm, b"y1", n * 4),
                "mask_x": _locate_raw_array(mm, b"mask_x", n),
                "mask_y": _locate_raw_array(mm, b"mask_y", n),
            }
            mm.close()
        self.offsets = offsets
        self.num_x = np.memmap(
            self.pickle_path,
            dtype=np.float32,
            mode="r",
            offset=offsets["num_x"],
            shape=(T, S, NUMERIC_FEATURE_COUNT),
        )
        self.cat_x = np.memmap(
            self.pickle_path,
            dtype=np.int64,
            mode="r",
            offset=offsets["cat_x"],
            shape=(T, S, CATEGORY_FEATURE_COUNT),
        )
        self.y1 = np.memmap(
            self.pickle_path,
            dtype=np.float32,
            mode="r",
            offset=offsets["y1"],
            shape=(T, S),
        )
        self.mask_x = np.memmap(
            self.pickle_path,
            dtype=np.bool_,
            mode="r",
            offset=offsets["mask_x"],
            shape=(T, S),
        )
        self.mask_y = np.memmap(
            self.pickle_path,
            dtype=np.bool_,
            mode="r",
            offset=offsets["mask_y"],
            shape=(T, S),
        )
        self.first_valid_x = np.argmax(self.mask_x, axis=0).astype(np.int32)
        self.validate()

    def validate(self) -> None:
        assert self.num_x.shape == (T, S, NUMERIC_FEATURE_COUNT)
        assert self.cat_x.shape == (T, S, CATEGORY_FEATURE_COUNT)
        assert self.y1.shape == self.mask_x.shape == self.mask_y.shape == (T, S)
        for start, stop in ((TRAIN_START, VALID_START), (VALID_START, TEST_START)):
            valid = (
                self.mask_x[start:stop]
                & self.mask_y[start:stop]
                & np.isfinite(self.y1[start:stop])
            )
            assert np.count_nonzero(valid) > 0
            assert not np.any(self.mask_y[start:stop] & ~self.mask_x[start:stop])
        assert not np.any(np.isfinite(self.y1[TEST_START:]))

    def eligible_stocks(self, time_idx: int, require_label: bool = True) -> np.ndarray:
        mask = np.asarray(self.mask_x[time_idx] & self.mask_y[time_idx])
        if require_label:
            mask &= np.isfinite(self.y1[time_idx])
        return np.flatnonzero(mask)


def deterministic_stock_sample(stocks: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or stocks.size <= cap:
        return stocks
    positions = np.linspace(0, stocks.size - 1, int(cap), dtype=np.int64)
    return stocks[positions]


def rank_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    usable = np.isfinite(predictions) & np.isfinite(labels)
    if np.count_nonzero(usable) < 2:
        return float("nan")
    prediction_ranks = rankdata(predictions[usable], method="average")
    label_ranks = rankdata(labels[usable], method="average")
    if np.std(prediction_ranks) <= 0 or np.std(label_ranks) <= 0:
        return float("nan")
    return float(np.corrcoef(prediction_ranks, label_ranks)[0, 1])


def summarize_ic(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "time_points": 0,
            "mean_rank_ic": float("nan"),
            "std_rank_ic": float("nan"),
            "icir": float("nan"),
            "min_rank_ic": float("nan"),
            "max_rank_ic": float("nan"),
        }
    std = float(array.std())
    mean = float(array.mean())
    return {
        "time_points": int(array.size),
        "mean_rank_ic": mean,
        "std_rank_ic": std,
        "icir": mean / std if std > 0 else float("nan"),
        "min_rank_ic": float(array.min()),
        "max_rank_ic": float(array.max()),
    }


def _column_rank_ic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    feature_ranks = rankdata(features, axis=0, method="average").astype(np.float64)
    label_ranks = rankdata(labels, method="average").astype(np.float64)
    feature_ranks -= feature_ranks.mean(axis=0)
    label_ranks -= label_ranks.mean()
    numerator = feature_ranks.T @ label_ranks
    denominator = np.sqrt(
        np.einsum("ij,ij->j", feature_ranks, feature_ranks)
        * np.dot(label_ranks, label_ranks)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(features.shape[1], np.nan),
        where=denominator > 0,
    )


def discover_numeric_features(
    data: CompetitionData,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    fold_matrices: dict[str, np.ndarray] = {}
    rows = []
    for fold in FOLDS:
        per_time = []
        for time_idx in range(fold.valid_start, fold.valid_stop):
            stocks = deterministic_stock_sample(
                data.eligible_stocks(time_idx, require_label=True),
                config.correlation_stock_cap,
            )
            features = np.asarray(data.num_x[time_idx, stocks], dtype=np.float64)
            labels = np.asarray(data.y1[time_idx, stocks], dtype=np.float64)
            per_time.append(_column_rank_ic(features, labels))
        matrix = np.vstack(per_time)
        fold_matrices[fold.name] = matrix
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        positives = np.nanmean(matrix > 0, axis=0)
        for index in range(NUMERIC_FEATURE_COUNT):
            rows.append(
                {
                    "fold": fold.name,
                    "feature_index": index,
                    "feature": f"num_{index}",
                    "mean_rank_ic": means[index],
                    "std_rank_ic": stds[index],
                    "icir": means[index] / stds[index] if stds[index] > 0 else np.nan,
                    "positive_rate": positives[index],
                }
            )

    details = pd.DataFrame(rows)
    pivot = details.pivot(index="feature_index", columns="fold", values="mean_rank_ic")
    summary = pd.DataFrame(index=np.arange(NUMERIC_FEATURE_COUNT))
    summary["feature"] = [f"num_{index}" for index in summary.index]
    summary["median_abs_fold_ic"] = np.median(np.abs(pivot.to_numpy()), axis=1)
    summary["mean_abs_fold_ic"] = np.mean(np.abs(pivot.to_numpy()), axis=1)
    summary["fold_sign_stability"] = np.maximum(
        np.mean(pivot.to_numpy() >= 0, axis=1),
        np.mean(pivot.to_numpy() < 0, axis=1),
    )
    for fold_name in pivot.columns:
        summary[f"{fold_name}_mean_ic"] = pivot[fold_name].to_numpy()

    sampled_times = np.linspace(
        TRAIN_START,
        VALID_START - 1,
        config.correlation_time_samples,
        dtype=int,
    )
    correlation_sum = np.zeros((NUMERIC_FEATURE_COUNT, NUMERIC_FEATURE_COUNT), dtype=np.float64)
    used_times = 0
    for time_idx in sampled_times:
        stocks = deterministic_stock_sample(
            data.eligible_stocks(time_idx, require_label=True),
            config.correlation_stock_cap,
        )
        if stocks.size < 2:
            continue
        features = np.asarray(data.num_x[time_idx, stocks], dtype=np.float64)
        ranks = rankdata(features, axis=0, method="average")
        correlation_sum += np.corrcoef(ranks, rowvar=False)
        used_times += 1
    mean_correlation = correlation_sum / max(used_times, 1)

    parent = np.arange(NUMERIC_FEATURE_COUNT)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return int(item)

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    upper_i, upper_j = np.triu_indices(NUMERIC_FEATURE_COUNT, k=1)
    for left, right, value in zip(upper_i, upper_j, mean_correlation[upper_i, upper_j]):
        if abs(value) >= 0.90:
            union(int(left), int(right))

    cluster_members: dict[int, list[int]] = {}
    for feature_index in range(NUMERIC_FEATURE_COUNT):
        cluster_members.setdefault(find(feature_index), []).append(feature_index)

    score = summary["median_abs_fold_ic"].to_numpy()
    representatives = {
        max(members, key=lambda index: score[index]) for members in cluster_members.values()
    }
    ranked = summary.sort_values(
        ["median_abs_fold_ic", "fold_sign_stability"],
        ascending=False,
    ).index.to_list()
    selected = [index for index in ranked if index in representatives]
    if len(selected) < config.top_feature_count:
        selected.extend(index for index in ranked if index not in selected)
    selected = np.asarray(selected[: config.top_feature_count], dtype=np.int32)
    history_features = selected[: config.history_feature_count].copy()

    cluster_id_by_feature = {index: find(index) for index in range(NUMERIC_FEATURE_COUNT)}
    summary["correlation_cluster"] = [
        cluster_id_by_feature[index] for index in summary.index
    ]
    summary["selected_top40"] = summary.index.isin(selected)
    summary["selected_history_top20"] = summary.index.isin(history_features)
    return summary.reset_index(names="feature_index"), selected, mean_correlation


def build_history_cache(
    data: CompetitionData,
    history_features: np.ndarray,
    config: PipelineConfig,
) -> dict[str, Path]:
    cache_dir = config.cache_dir_obj
    signature = "-".join(map(str, history_features.tolist()))
    manifest_path = cache_dir / "history_cache_manifest.json"
    paths = {
        "cumsum": cache_dir / "history_cumsum.dat",
        "cumsq": cache_dir / "history_cumsq.dat",
        "cumcount": cache_dir / "history_cumcount.dat",
        "cross_rank": cache_dir / "history_cross_rank.dat",
        "group_features": cache_dir / "history_group_features.dat",
    }
    expected = {
        "signature": signature,
        "shape": [T, S, int(history_features.size)],
    }
    if manifest_path.exists() and all(path.exists() for path in paths.values()):
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current == expected:
            return paths

    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_count = int(history_features.size)
    cumsum = np.memmap(
        paths["cumsum"],
        dtype=np.float32,
        mode="w+",
        shape=(T + 1, S, feature_count),
    )
    cumsq = np.memmap(
        paths["cumsq"],
        dtype=np.float32,
        mode="w+",
        shape=(T + 1, S, feature_count),
    )
    cumcount = np.memmap(
        paths["cumcount"],
        dtype=np.uint16,
        mode="w+",
        shape=(T + 1, S),
    )
    cross_rank = np.memmap(
        paths["cross_rank"],
        dtype=np.float32,
        mode="w+",
        shape=(T, S, feature_count),
    )
    group_features = np.memmap(
        paths["group_features"],
        dtype=np.float32,
        mode="w+",
        shape=(T, S, 60),
    )
    cumsum[0] = 0
    cumsq[0] = 0
    cumcount[0] = 0

    log("Building reusable history, rank, and category-group caches", config.verbose)
    top10 = history_features[:10]
    for time_idx in range(T):
        valid = np.asarray(data.mask_x[time_idx])
        values = np.asarray(
            data.num_x[time_idx][:, history_features],
            dtype=np.float32,
        )
        valid_values = np.where(valid[:, None], values, 0.0)
        cumsum[time_idx + 1] = cumsum[time_idx] + valid_values
        cumsq[time_idx + 1] = cumsq[time_idx] + valid_values * valid_values
        cumcount[time_idx + 1] = cumcount[time_idx] + valid.astype(np.uint16)

        ranks = np.full((S, feature_count), np.nan, dtype=np.float32)
        if np.count_nonzero(valid) >= 2:
            valid_ranks = rankdata(values[valid], axis=0, method="average")
            valid_ranks = (
                valid_ranks - 1.0
            ) / max(np.count_nonzero(valid) - 1, 1) - 0.5
            ranks[valid] = valid_ranks.astype(np.float32)
        cross_rank[time_idx] = ranks

        top_values = np.asarray(data.num_x[time_idx][:, top10], dtype=np.float32)
        output_offset = 0
        for category_index in (1, 6):
            categories = np.asarray(data.cat_x[time_idx, :, category_index], dtype=np.int64)
            local_output = np.full((S, 30), np.nan, dtype=np.float32)
            for category in np.unique(categories[valid]):
                members = valid & (categories == category)
                member_values = top_values[members]
                if member_values.shape[0] == 0:
                    continue
                medians = np.median(member_values, axis=0)
                means = member_values.mean(axis=0)
                stds = member_values.std(axis=0)
                within_rank = rankdata(member_values, axis=0, method="average")
                within_rank = (
                    within_rank - 1.0
                ) / max(member_values.shape[0] - 1, 1) - 0.5
                group_block = np.concatenate(
                    [
                        within_rank,
                        member_values - medians,
                        (member_values - means) / np.maximum(stds, 1e-6),
                    ],
                    axis=1,
                )
                local_output[members] = group_block.astype(np.float32)
            group_features[time_idx, :, output_offset : output_offset + 30] = local_output
            output_offset += 30

        if (time_idx + 1) % 300 == 0 or time_idx == T - 1:
            log(f"History cache {time_idx + 1}/{T}", config.verbose)

    for array in (cumsum, cumsq, cumcount, cross_rank, group_features):
        array.flush()
    del cumsum, cumsq, cumcount, cross_rank, group_features
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def open_history_cache(
    paths: dict[str, Path],
    history_feature_count: int,
) -> dict[str, np.memmap]:
    return {
        "cumsum": np.memmap(
            paths["cumsum"],
            dtype=np.float32,
            mode="r",
            shape=(T + 1, S, history_feature_count),
        ),
        "cumsq": np.memmap(
            paths["cumsq"],
            dtype=np.float32,
            mode="r",
            shape=(T + 1, S, history_feature_count),
        ),
        "cumcount": np.memmap(
            paths["cumcount"],
            dtype=np.uint16,
            mode="r",
            shape=(T + 1, S),
        ),
        "cross_rank": np.memmap(
            paths["cross_rank"],
            dtype=np.float32,
            mode="r",
            shape=(T, S, history_feature_count),
        ),
        "group_features": np.memmap(
            paths["group_features"],
            dtype=np.float32,
            mode="r",
            shape=(T, S, 60),
        ),
    }


def _interaction_values(
    data: CompetitionData,
    time_idx: int,
    stocks: np.ndarray,
    specifications: Sequence[dict],
) -> np.ndarray:
    if not specifications:
        return np.empty((stocks.size, 0), dtype=np.float32)
    needed = sorted(
        {
            int(specification[key])
            for specification in specifications
            for key in ("left", "right")
        }
    )
    valid = np.asarray(data.mask_x[time_idx])
    valid_stocks = np.flatnonzero(valid)
    all_values = np.asarray(data.num_x[time_idx, valid_stocks][:, needed], dtype=np.float64)
    all_ranks = rankdata(all_values, axis=0, method="average")
    all_ranks = (all_ranks - 1.0) / max(valid_stocks.size - 1, 1) - 0.5
    position_by_stock = np.full(S, -1, dtype=np.int32)
    position_by_stock[valid_stocks] = np.arange(valid_stocks.size, dtype=np.int32)
    requested_positions = position_by_stock[stocks]
    index_by_feature = {feature: index for index, feature in enumerate(needed)}
    selected_values = all_values[requested_positions]
    selected_ranks = all_ranks[requested_positions]
    outputs = []
    for specification in specifications:
        left_position = index_by_feature[int(specification["left"])]
        right_position = index_by_feature[int(specification["right"])]
        transform = specification["transform"]
        left_rank = selected_ranks[:, left_position]
        right_rank = selected_ranks[:, right_position]
        if transform == "rank_product":
            value = left_rank * right_rank
        elif transform == "rank_difference":
            value = left_rank - right_rank
        elif transform == "rank_abs_difference":
            value = np.abs(left_rank - right_rank)
        elif transform == "robust_product":
            left_value = np.clip(selected_values[:, left_position], -5.0, 5.0)
            right_value = np.clip(selected_values[:, right_position], -5.0, 5.0)
            value = left_value * right_value
        else:
            raise ValueError(f"Unknown interaction transform {transform}")
        outputs.append(np.asarray(value, dtype=np.float32))
    return np.column_stack(outputs).astype(np.float32, copy=False)


def screen_interactions(
    data: CompetitionData,
    numeric_summary: pd.DataFrame,
    selected_features: np.ndarray,
    mean_correlation: np.ndarray,
    config: PipelineConfig,
) -> tuple[list[dict], pd.DataFrame]:
    strong = selected_features[:20]
    supplementary = selected_features[20:40]
    feature_score = numeric_summary.set_index("feature_index")[
        "median_abs_fold_ic"
    ].to_dict()
    pair_candidates = []
    for left_position, left in enumerate(strong):
        for right in strong[left_position + 1 :]:
            if np.isfinite(mean_correlation[left, right]) and abs(mean_correlation[left, right]) < 0.85:
                pair_candidates.append((int(left), int(right)))
        for right in supplementary:
            if np.isfinite(mean_correlation[left, right]) and abs(mean_correlation[left, right]) < 0.85:
                pair_candidates.append((int(left), int(right)))
    pair_candidates = sorted(
        set(pair_candidates),
        key=lambda pair: (
            feature_score[pair[0]] + feature_score[pair[1]]
        )
        * (1.0 - abs(mean_correlation[pair[0], pair[1]])),
        reverse=True,
    )[: config.interaction_pair_count]
    specifications = [
        {"left": left, "right": right, "transform": transform}
        for left, right in pair_candidates
        for transform in (
            "rank_product",
            "rank_difference",
            "rank_abs_difference",
            "robust_product",
        )
    ]

    fold_scores: dict[str, list[float]] = {}
    for fold in FOLDS:
        train_times = np.linspace(
            fold.train_start,
            fold.train_stop - 1,
            config.discovery_train_time_samples,
            dtype=int,
        )
        train_features = []
        train_labels = []
        for time_idx in train_times:
            stocks = deterministic_stock_sample(
                data.eligible_stocks(time_idx, require_label=True),
                config.discovery_stock_cap,
            )
            current = np.asarray(
                data.num_x[time_idx, stocks][:, selected_features],
                dtype=np.float32,
            )
            current_ranks = rankdata(current[:, :20], axis=0, method="average")
            current_ranks = (
                (current_ranks - 1.0) / max(stocks.size - 1, 1) - 0.5
            ).astype(np.float32)
            train_features.append(np.column_stack([current, current_ranks]))
            train_labels.append(np.asarray(data.y1[time_idx, stocks], dtype=np.float32))
        ridge = Ridge(alpha=10.0)
        ridge.fit(np.vstack(train_features), np.concatenate(train_labels))

        per_time_scores = []
        for time_idx in range(fold.valid_start, fold.valid_stop):
            stocks = deterministic_stock_sample(
                data.eligible_stocks(time_idx, require_label=True),
                config.correlation_stock_cap,
            )
            current = np.asarray(
                data.num_x[time_idx, stocks][:, selected_features],
                dtype=np.float32,
            )
            current_ranks = rankdata(current[:, :20], axis=0, method="average")
            current_ranks = (
                (current_ranks - 1.0) / max(stocks.size - 1, 1) - 0.5
            ).astype(np.float32)
            base = np.column_stack([current, current_ranks])
            residual = np.asarray(data.y1[time_idx, stocks]) - ridge.predict(base)
            interactions = _interaction_values(data, time_idx, stocks, specifications)
            per_time_scores.append(_column_rank_ic(interactions, residual))
        fold_scores[fold.name] = np.nanmean(np.vstack(per_time_scores), axis=0).tolist()

    rows = []
    for index, specification in enumerate(specifications):
        values = np.asarray([fold_scores[fold.name][index] for fold in FOLDS])
        rows.append(
            {
                **specification,
                **{fold.name: fold_scores[fold.name][index] for fold in FOLDS},
                "median_abs_residual_ic": float(np.median(np.abs(values))),
                "sign_stability": float(
                    max(np.mean(values >= 0), np.mean(values < 0))
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["sign_stability", "median_abs_residual_ic"],
        ascending=False,
    )
    selected_table = table[
        table["sign_stability"] >= (2.0 / 3.0)
    ].head(config.interaction_keep_count)
    selected_specifications = [
        {
            "left": int(row.left),
            "right": int(row.right),
            "transform": str(row.transform),
        }
        for row in selected_table.itertuples(index=False)
    ]
    return selected_specifications, table


@dataclass
class CategoryState:
    mappings: list[np.ndarray]
    unknown_codes: list[int]
    cat5_frequency: np.ndarray
    cat5_target_sum: np.ndarray
    cat5_target_count: np.ndarray
    cat5_global_target: float
    clip_low: np.ndarray
    clip_high: np.ndarray
    train_start: int
    train_stop: int


def fit_category_state(
    data: CompetitionData,
    train_start: int,
    train_stop: int,
    history_features: np.ndarray,
    config: PipelineConfig,
) -> CategoryState:
    maxima = np.zeros(CATEGORY_FEATURE_COUNT, dtype=np.int64)
    seen_sets = [set() for _ in range(CATEGORY_FEATURE_COUNT)]
    cat5_max = int(np.max(data.cat_x[:, :, 5]))
    cat5_frequency = np.zeros(cat5_max + 1, dtype=np.int64)
    cat5_target_sum = np.zeros(cat5_max + 1, dtype=np.float64)
    cat5_target_count = np.zeros(cat5_max + 1, dtype=np.int64)
    total_target_sum = 0.0
    total_target_count = 0
    sampled_numeric = []

    sample_times = set(
        np.linspace(train_start, train_stop - 1, 64, dtype=int).tolist()
    )
    for time_idx in range(train_start, train_stop):
        stocks = data.eligible_stocks(time_idx, require_label=True)
        categories = np.asarray(data.cat_x[time_idx, stocks], dtype=np.int64)
        labels = np.asarray(data.y1[time_idx, stocks], dtype=np.float64)
        for category_index in range(CATEGORY_FEATURE_COUNT):
            values = categories[:, category_index]
            maxima[category_index] = max(maxima[category_index], int(values.max()))
            seen_sets[category_index].update(np.unique(values).tolist())
        cat5 = categories[:, 5]
        cat5_frequency += np.bincount(cat5, minlength=cat5_max + 1)
        cat5_target_sum += np.bincount(cat5, weights=labels, minlength=cat5_max + 1)
        cat5_target_count += np.bincount(cat5, minlength=cat5_max + 1)
        total_target_sum += float(labels.sum())
        total_target_count += int(labels.size)
        if time_idx in sample_times:
            sampled_stocks = deterministic_stock_sample(stocks, config.correlation_stock_cap)
            sampled_numeric.append(
                np.asarray(
                    data.num_x[time_idx, sampled_stocks][:, history_features],
                    dtype=np.float32,
                )
            )

    mappings = []
    unknown_codes = []
    for category_index, seen in enumerate(seen_sets):
        maximum = max(maxima[category_index], 0)
        mapping = np.full(maximum + 1, -1, dtype=np.int32)
        for encoded, original in enumerate(sorted(seen)):
            mapping[int(original)] = encoded
        unknown = len(seen)
        mappings.append(mapping)
        unknown_codes.append(unknown)

    sampled = np.vstack(sampled_numeric)
    clip_low = np.quantile(sampled, 0.005, axis=0).astype(np.float32)
    clip_high = np.quantile(sampled, 0.995, axis=0).astype(np.float32)
    return CategoryState(
        mappings=mappings,
        unknown_codes=unknown_codes,
        cat5_frequency=cat5_frequency,
        cat5_target_sum=cat5_target_sum,
        cat5_target_count=cat5_target_count,
        cat5_global_target=total_target_sum / max(total_target_count, 1),
        clip_low=clip_low,
        clip_high=clip_high,
        train_start=train_start,
        train_stop=train_stop,
    )


def encode_categories(raw: np.ndarray, state: CategoryState) -> np.ndarray:
    encoded = np.empty(raw.shape, dtype=np.float32)
    for category_index in range(CATEGORY_FEATURE_COUNT):
        mapping = state.mappings[category_index]
        values = raw[:, category_index]
        result = np.full(values.shape, state.unknown_codes[category_index], dtype=np.int32)
        in_range = (values >= 0) & (values < mapping.size)
        mapped = mapping[values[in_range]]
        mapped[mapped < 0] = state.unknown_codes[category_index]
        result[in_range] = mapped
        encoded[:, category_index] = result
    return encoded


@dataclass
class FeatureMatrix:
    path: Path
    shape: tuple[int, int]
    y: np.ndarray
    relevance: np.ndarray
    groups: np.ndarray
    times: np.ndarray
    feature_names: list[str]
    categorical_names: list[str]
    block_ends: dict[str, int]

    def open(self) -> np.memmap:
        return np.memmap(
            self.path,
            dtype=np.float32,
            mode="r",
            shape=self.shape,
        )


def feature_layout(
    selected_features: np.ndarray,
    history_features: np.ndarray,
    interaction_specs: Sequence[dict],
) -> tuple[list[str], list[str], dict[str, int]]:
    names = [f"num_{index}" for index in selected_features]
    names += [f"rank_num_{index}" for index in history_features]
    names += [
        f"lag_{lag}_num_{index}"
        for lag in (1, 5, 20, 60)
        for index in history_features
    ]
    names += [
        f"roll_{stat}_{window}_num_{index}"
        for window in (5, 20, 60)
        for stat in ("mean", "std", "change")
        for index in history_features
    ]
    names += [f"lag_{lag}_available" for lag in (1, 5, 20, 60)]
    names += [f"history_coverage_{window}" for window in (5, 20, 60)]
    names += ["stock_age"]
    numeric_end = len(names)

    category_names = [f"cat_{index}" for index in range(CATEGORY_FEATURE_COUNT)]
    names += category_names
    raw_category_end = len(names)

    names += ["cat_5_frequency", "cat_5_log_frequency", "cat_5_past_target_mean"]
    names += [
        f"cat_{category}_group_{stat}_num_{index}"
        for category in (1, 6)
        for stat in ("rank", "median_diff", "zscore")
        for index in history_features[:10]
    ]
    category_extended_end = len(names)

    names += [
        f"{spec['transform']}_num_{spec['left']}_num_{spec['right']}"
        for spec in interaction_specs
    ]
    interaction_end = len(names)
    return (
        names,
        category_names,
        {
            "numeric": numeric_end,
            "raw_category": raw_category_end,
            "category_extended": category_extended_end,
            "interaction": interaction_end,
        },
    )


def _trend_estimate(
    data: CompetitionData,
    time_idx: int,
    stocks: np.ndarray,
    history_features: np.ndarray,
    state: CategoryState,
) -> np.ndarray:
    start = max(0, time_idx - 20)
    if start >= time_idx:
        return np.full((stocks.size, history_features.size), np.nan, dtype=np.float32)
    values = np.asarray(
        data.num_x[start:time_idx][:, stocks][:, :, history_features],
        dtype=np.float32,
    )
    valid = np.asarray(data.mask_x[start:time_idx][:, stocks], dtype=bool)
    step_count = values.shape[0]
    ages = np.arange(step_count - 1, -1, -1, dtype=np.float32)
    weights = np.exp(-math.log(2.0) * ages / 5.0)[:, None, None]
    weighted_valid = weights * valid[:, :, None]
    denominator = weighted_valid.sum(axis=0)
    ewma = np.divide(
        (values * weighted_valid).sum(axis=0),
        denominator,
        out=np.full((stocks.size, history_features.size), np.nan, dtype=np.float32),
        where=denominator > 0,
    )
    last = np.full_like(ewma, np.nan)
    for local_time in range(step_count - 1, -1, -1):
        unresolved = ~np.isfinite(last[:, 0]) & valid[local_time]
        if np.any(unresolved):
            last[unresolved] = values[local_time, unresolved]
    if step_count >= 2:
        pair_valid = valid[1:] & valid[:-1]
        differences = values[1:] - values[:-1]
        drift = np.divide(
            (differences * pair_valid[:, :, None]).sum(axis=0),
            pair_valid.sum(axis=0)[:, None],
            out=np.zeros_like(ewma),
            where=pair_valid.sum(axis=0)[:, None] > 0,
        )
    else:
        drift = np.zeros_like(ewma)
    count = valid.sum(axis=0)
    gap = np.zeros(stocks.size, dtype=np.float32)
    for stock_position in range(stocks.size):
        valid_positions = np.flatnonzero(valid[:, stock_position])
        gap[stock_position] = (
            step_count - 1 - valid_positions[-1] if valid_positions.size else 20
        )
    reliability = (count / 20.0) * np.exp(-gap / 20.0)
    estimate = 0.5 * last + 0.5 * ewma + 0.5 * reliability[:, None] * drift
    estimate = np.clip(estimate, state.clip_low, state.clip_high)
    estimate[count == 0] = np.nan
    return estimate.astype(np.float32)


def build_feature_matrix(
    data: CompetitionData,
    history_cache: dict[str, np.memmap],
    selected_features: np.ndarray,
    history_features: np.ndarray,
    interaction_specs: Sequence[dict],
    category_state: CategoryState,
    start: int,
    stop: int,
    stock_cap: int | None,
    role: str,
    imputation_mode: str,
    matrix_name: str,
    config: PipelineConfig,
) -> FeatureMatrix:
    feature_names, categorical_names, block_ends = feature_layout(
        selected_features,
        history_features,
        interaction_specs,
    )
    stocks_by_time = []
    group_sizes = []
    for time_idx in range(start, stop):
        stocks = deterministic_stock_sample(
            data.eligible_stocks(time_idx, require_label=(role != "test")),
            stock_cap,
        )
        stocks_by_time.append(stocks)
        group_sizes.append(int(stocks.size))
    row_count = int(sum(group_sizes))
    feature_count = len(feature_names)
    matrix_path = config.cache_dir_obj / f"{matrix_name}.dat"
    matrix = np.memmap(
        matrix_path,
        dtype=np.float32,
        mode="w+",
        shape=(row_count, feature_count),
    )
    labels = np.full(row_count, np.nan, dtype=np.float32)
    times = np.empty(row_count, dtype=np.int32)

    if role == "train":
        running_sum = np.zeros_like(category_state.cat5_target_sum)
        running_count = np.zeros_like(category_state.cat5_target_count)
    else:
        running_sum = category_state.cat5_target_sum.copy()
        running_count = category_state.cat5_target_count.copy()

    offset = 0
    history_count = int(history_features.size)
    for local_index, (time_idx, stocks) in enumerate(
        zip(range(start, stop), stocks_by_time)
    ):
        count = stocks.size
        if count == 0:
            continue
        blocks = []
        current = np.asarray(
            data.num_x[time_idx, stocks][:, selected_features],
            dtype=np.float32,
        )
        blocks.append(current)
        blocks.append(np.asarray(history_cache["cross_rank"][time_idx, stocks]))

        trend_estimate = None
        if imputation_mode == "trend":
            trend_estimate = _trend_estimate(
                data,
                time_idx,
                stocks,
                history_features,
                category_state,
            )

        lag_availability = []
        for lag in (1, 5, 20, 60):
            if time_idx - lag < 0:
                lag_values = np.full((count, history_count), np.nan, dtype=np.float32)
                available = np.zeros(count, dtype=np.float32)
            else:
                available_mask = np.asarray(data.mask_x[time_idx - lag, stocks])
                lag_values = np.asarray(
                    data.num_x[time_idx - lag, stocks][:, history_features],
                    dtype=np.float32,
                )
                lag_values[~available_mask] = np.nan
                available = available_mask.astype(np.float32)
            if trend_estimate is not None:
                missing = ~np.isfinite(lag_values)
                lag_values[missing] = np.broadcast_to(
                    trend_estimate,
                    lag_values.shape,
                )[missing]
            blocks.append(lag_values)
            lag_availability.append(available)

        coverage_columns = []
        for window in (5, 20, 60):
            window_start = max(0, time_idx - window)
            sample_count = (
                history_cache["cumcount"][time_idx, stocks]
                - history_cache["cumcount"][window_start, stocks]
            ).astype(np.float32)
            sums = (
                history_cache["cumsum"][time_idx, stocks]
                - history_cache["cumsum"][window_start, stocks]
            )
            squared_sums = (
                history_cache["cumsq"][time_idx, stocks]
                - history_cache["cumsq"][window_start, stocks]
            )
            mean = np.divide(
                sums,
                sample_count[:, None],
                out=np.full_like(sums, np.nan),
                where=sample_count[:, None] > 0,
            )
            variance = np.divide(
                squared_sums,
                sample_count[:, None],
                out=np.full_like(squared_sums, np.nan),
                where=sample_count[:, None] > 0,
            ) - mean * mean
            std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
            change = np.full_like(mean, np.nan)
            usable_change = sample_count > 1
            if np.any(usable_change):
                selected_stocks = stocks[usable_change]
                first_times = (time_idx - sample_count[usable_change]).astype(np.int32)
                first_values = np.asarray(
                    data.num_x[first_times, selected_stocks][:, history_features],
                    dtype=np.float32,
                )
                last_values = np.asarray(
                    data.num_x[time_idx - 1, selected_stocks][:, history_features],
                    dtype=np.float32,
                )
                change[usable_change] = (
                    last_values - first_values
                ) / (sample_count[usable_change, None] - 1.0)
            blocks.extend([mean.astype(np.float32), std, change.astype(np.float32)])
            coverage_columns.append(sample_count / float(window))

        first_valid = data.first_valid_x[stocks]
        stock_age = np.maximum(time_idx - first_valid, 0).astype(np.float32)
        blocks.append(np.column_stack(lag_availability).astype(np.float32))
        blocks.append(np.column_stack(coverage_columns).astype(np.float32))
        blocks.append(stock_age[:, None])

        raw_categories = np.asarray(data.cat_x[time_idx, stocks], dtype=np.int64)
        encoded_categories = encode_categories(raw_categories, category_state)
        blocks.append(encoded_categories)

        cat5 = raw_categories[:, 5]
        frequency = np.zeros(count, dtype=np.float32)
        in_range = (cat5 >= 0) & (cat5 < category_state.cat5_frequency.size)
        frequency[in_range] = category_state.cat5_frequency[cat5[in_range]]
        target_mean = np.full(count, category_state.cat5_global_target, dtype=np.float32)
        if np.any(in_range):
            sums = running_sum[cat5[in_range]]
            counts = running_count[cat5[in_range]]
            target_mean[in_range] = (
                (sums + 20.0 * category_state.cat5_global_target)
                / (counts + 20.0)
            ).astype(np.float32)
        blocks.append(
            np.column_stack(
                [frequency, np.log1p(frequency), target_mean]
            ).astype(np.float32)
        )
        blocks.append(np.asarray(history_cache["group_features"][time_idx, stocks]))
        blocks.append(_interaction_values(data, time_idx, stocks, interaction_specs))

        row_block = np.column_stack(blocks).astype(np.float32, copy=False)
        if row_block.shape[1] != feature_count:
            raise AssertionError(
                f"Feature count mismatch {row_block.shape[1]} != {feature_count}"
            )
        matrix[offset : offset + count] = row_block
        times[offset : offset + count] = time_idx
        if role != "test":
            current_labels = np.asarray(data.y1[time_idx, stocks], dtype=np.float32)
            labels[offset : offset + count] = current_labels
            if role == "train":
                current_cat5 = raw_categories[:, 5]
                running_sum += np.bincount(
                    current_cat5,
                    weights=current_labels,
                    minlength=running_sum.size,
                )
                running_count += np.bincount(
                    current_cat5,
                    minlength=running_count.size,
                )
        offset += count
        if (local_index + 1) % 100 == 0 or time_idx == stop - 1:
            log(
                f"{matrix_name}: {local_index + 1}/{stop - start} time points",
                config.verbose,
            )
    matrix.flush()
    del matrix
    if role == "test":
        relevance = np.zeros(row_count, dtype=np.int32)
    else:
        relevance = np.minimum(
            np.floor(np.clip(labels, 0.0, 1.0) * 64.0),
            63,
        ).astype(np.int32)
    return FeatureMatrix(
        path=matrix_path,
        shape=(row_count, feature_count),
        y=labels,
        relevance=relevance,
        groups=np.asarray(group_sizes, dtype=np.int32),
        times=times,
        feature_names=feature_names,
        categorical_names=categorical_names,
        block_ends=block_ends,
    )


def mean_rank_ic_from_groups(
    predictions: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, list[float]]:
    values = []
    offset = 0
    for group_size in groups:
        group_size = int(group_size)
        if group_size > 1:
            values.append(
                rank_ic(
                    predictions[offset : offset + group_size],
                    labels[offset : offset + group_size],
                )
            )
        offset += group_size
    summary = summarize_ic(values)
    return float(summary["mean_rank_ic"]), values


def base_lgb_params() -> dict:
    return {
        "objective": "lambdarank",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "label_gain": list(range(64)),
        "lambdarank_truncation_level": 1024,
        "verbosity": -1,
        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
        "num_threads": max(1, (os.cpu_count() or 8) - 2),
    }


def _categorical_names_for_prefix(matrix: FeatureMatrix, prefix: int) -> list[str]:
    available = set(matrix.feature_names[:prefix])
    return [name for name in matrix.categorical_names if name in available]


def train_ranker(
    train_matrix: FeatureMatrix,
    valid_matrix: FeatureMatrix,
    prefix: int,
    params: dict,
    max_rounds: int,
    early_stopping_rounds: int,
    verbose: bool = False,
) -> tuple[lgb.Booster, dict]:
    train_values = train_matrix.open()
    valid_values = valid_matrix.open()
    feature_names = train_matrix.feature_names[:prefix]
    categorical_names = _categorical_names_for_prefix(train_matrix, prefix)
    train_set = lgb.Dataset(
        train_values[:, :prefix],
        label=train_matrix.relevance,
        group=train_matrix.groups,
        feature_name=feature_names,
        categorical_feature=categorical_names,
        params={"max_bin": int(params.get("max_bin", 255))},
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_values[:, :prefix],
        label=valid_matrix.relevance,
        group=valid_matrix.groups,
        feature_name=feature_names,
        categorical_feature=categorical_names,
        reference=train_set,
        free_raw_data=False,
    )

    def custom_metric(predictions, _dataset):
        mean_ic, _ = mean_rank_ic_from_groups(
            predictions,
            valid_matrix.y,
            valid_matrix.groups,
        )
        return "mean_rank_ic", mean_ic, True

    callbacks = [
        lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=verbose),
        lgb.log_evaluation(50 if verbose else 0),
    ]
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        valid_names=["validation"],
        feval=custom_metric,
        callbacks=callbacks,
    )
    predictions = booster.predict(
        valid_values[:, :prefix],
        num_iteration=booster.best_iteration,
    )
    _, ic_values = mean_rank_ic_from_groups(
        predictions,
        valid_matrix.y,
        valid_matrix.groups,
    )
    metrics = summarize_ic(ic_values)
    metrics["best_iteration"] = int(booster.best_iteration)
    del train_set, valid_set, train_values, valid_values
    gc.collect()
    return booster, metrics


def train_ranker_no_validation(
    train_matrix: FeatureMatrix,
    prefix: int,
    params: dict,
    boost_rounds: int,
) -> lgb.Booster:
    values = train_matrix.open()
    feature_names = train_matrix.feature_names[:prefix]
    categorical_names = _categorical_names_for_prefix(train_matrix, prefix)
    dataset = lgb.Dataset(
        values[:, :prefix],
        label=train_matrix.relevance,
        group=train_matrix.groups,
        feature_name=feature_names,
        categorical_feature=categorical_names,
        params={"max_bin": int(params.get("max_bin", 255))},
        free_raw_data=False,
    )
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=int(boost_rounds),
        callbacks=[lgb.log_evaluation(0)],
    )
    del dataset, values
    gc.collect()
    return booster


def remove_feature_matrix(matrix: FeatureMatrix) -> None:
    if matrix.path.exists():
        matrix.path.unlink()


def imputation_reconstruction_diagnostic(
    data: CompetitionData,
    history_features: np.ndarray,
    state: CategoryState,
    config: PipelineConfig,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(config.seed)
    candidate_times = np.linspace(TRAIN_START + 80, VALID_START - 1, 96, dtype=int)
    for gap in (1, 5, 20):
        neutral_errors = []
        trend_errors = []
        for time_idx in candidate_times:
            stocks = data.eligible_stocks(time_idx, require_label=True)
            stocks = stocks[data.first_valid_x[stocks] <= time_idx - 80]
            if stocks.size == 0:
                continue
            take = min(120, stocks.size)
            stocks = np.sort(rng.choice(stocks, size=take, replace=False))
            history_stop = time_idx - gap + 1
            estimate = _trend_estimate(
                data,
                history_stop,
                stocks,
                history_features,
                state,
            )
            target = np.asarray(
                data.num_x[time_idx, stocks][:, history_features],
                dtype=np.float32,
            )
            trend_errors.append(np.nanmean(np.abs(estimate - target)))
            neutral_errors.append(np.mean(np.abs(target)))
        rows.append(
            {
                "gap": gap,
                "neutral_zero_mae": float(np.mean(neutral_errors)),
                "causal_ewma_trend_mae": float(np.mean(trend_errors)),
            }
        )
    return pd.DataFrame(rows)


def run_feature_stage_ablation(
    data: CompetitionData,
    history_cache: dict[str, np.memmap],
    selected_features: np.ndarray,
    history_features: np.ndarray,
    interaction_specs: Sequence[dict],
    imputation_mode: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, str]:
    stages = ("numeric", "raw_category", "category_extended", "interaction")
    rows = []
    params = base_lgb_params()
    params.update(
        {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 250,
        }
    )
    for fold in FOLDS:
        state = fit_category_state(
            data,
            fold.train_start,
            fold.train_stop,
            history_features,
            config,
        )
        train_matrix = build_feature_matrix(
            data,
            history_cache,
            selected_features,
            history_features,
            interaction_specs,
            state,
            fold.train_start,
            fold.train_stop,
            config.ablation_stock_cap,
            "train",
            imputation_mode,
            f"ablation_{fold.name}_train",
            config,
        )
        valid_matrix = build_feature_matrix(
            data,
            history_cache,
            selected_features,
            history_features,
            interaction_specs,
            state,
            fold.valid_start,
            fold.valid_stop,
            config.ablation_stock_cap,
            "valid",
            imputation_mode,
            f"ablation_{fold.name}_valid",
            config,
        )
        for stage in stages:
            booster, metrics = train_ranker(
                train_matrix,
                valid_matrix,
                train_matrix.block_ends[stage],
                params,
                max_rounds=700,
                early_stopping_rounds=50,
                verbose=False,
            )
            rows.append({"fold": fold.name, "stage": stage, **metrics})
            del booster
            gc.collect()
        remove_feature_matrix(train_matrix)
        remove_feature_matrix(valid_matrix)

    table = pd.DataFrame(rows)
    pivot = table.pivot(index="fold", columns="stage", values="mean_rank_ic")
    predecessor = {
        "numeric": None,
        "raw_category": "numeric",
        "category_extended": "raw_category",
        "interaction": "category_extended",
    }
    table["relative_delta"] = [
        0.0
        if predecessor[stage] is None
        else float(pivot.loc[fold, stage] - pivot.loc[fold, predecessor[stage]])
        for fold, stage in zip(table["fold"], table["stage"])
    ]
    stage_summary = (
        table.groupby("stage", as_index=False)
        .agg(
            median_rank_ic=("mean_rank_ic", "median"),
            mean_rank_ic=("mean_rank_ic", "mean"),
            std_across_folds=("mean_rank_ic", "std"),
            median_relative_delta=("relative_delta", "median"),
            mean_relative_delta=("relative_delta", "mean"),
            positive_folds=("relative_delta", lambda values: int(np.count_nonzero(values > 0))),
        )
    )
    best_stage = "numeric"
    raw_gain = pivot["raw_category"] - pivot["numeric"]
    if np.count_nonzero(raw_gain > 0) >= 2 and float(np.median(raw_gain)) > 0:
        best_stage = "raw_category"
    extended_gain = pivot["category_extended"] - pivot[best_stage]
    if np.count_nonzero(extended_gain > 0) >= 2 and float(np.median(extended_gain)) > 0:
        best_stage = "category_extended"
    interaction_gain = pivot["interaction"] - pivot[best_stage]
    if (
        np.count_nonzero(interaction_gain > 0) >= 2
        and float(np.median(interaction_gain)) >= config.interaction_ic_threshold
    ):
        best_stage = "interaction"
    return table.merge(stage_summary, on="stage", suffixes=("", "_stage")), best_stage


def compare_imputation_modes(
    data: CompetitionData,
    history_cache: dict[str, np.memmap],
    selected_features: np.ndarray,
    history_features: np.ndarray,
    interaction_specs: Sequence[dict],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, str]:
    rows = []
    params = base_lgb_params()
    params.update({"learning_rate": 0.05, "num_leaves": 63})
    for fold in FOLDS:
        state = fit_category_state(
            data,
            fold.train_start,
            fold.train_stop,
            history_features,
            config,
        )
        for mode in ("native", "trend"):
            train_matrix = build_feature_matrix(
                data,
                history_cache,
                selected_features,
                history_features,
                interaction_specs,
                state,
                fold.train_start,
                fold.train_stop,
                config.ablation_stock_cap,
                "train",
                mode,
                f"imputation_{fold.name}_{mode}_train",
                config,
            )
            valid_matrix = build_feature_matrix(
                data,
                history_cache,
                selected_features,
                history_features,
                interaction_specs,
                state,
                fold.valid_start,
                fold.valid_stop,
                config.ablation_stock_cap,
                "valid",
                mode,
                f"imputation_{fold.name}_{mode}_valid",
                config,
            )
            booster, metrics = train_ranker(
                train_matrix,
                valid_matrix,
                train_matrix.block_ends["numeric"],
                params,
                max_rounds=700,
                early_stopping_rounds=50,
                verbose=False,
            )
            rows.append({"fold": fold.name, "mode": mode, **metrics})
            del booster
            remove_feature_matrix(train_matrix)
            remove_feature_matrix(valid_matrix)
            gc.collect()
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="fold", columns="mode", values="mean_rank_ic")
    gains = pivot["trend"] - pivot["native"]
    selected_mode = (
        "trend"
        if (
            np.count_nonzero(gains > 0) >= 2
            and float(np.median(gains)) >= config.imputation_ic_threshold
        )
        else "native"
    )
    return table, selected_mode


def _sample_trial_params(trial: optuna.Trial, seed: int) -> dict:
    params = base_lgb_params()
    params.update(
        {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127, step=16),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 100, 1000, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.60, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.60, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 30.0, log=True),
            "max_bin": trial.suggest_categorical("max_bin", [127, 255]),
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
        }
    )
    return params


def tune_parameters(
    train_matrix: FeatureMatrix,
    valid_matrix: FeatureMatrix,
    stage: str,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, list[dict]]:
    prefix = train_matrix.block_ends[stage]
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = _sample_trial_params(trial, config.seed)
        booster, metrics = train_ranker(
            train_matrix,
            valid_matrix,
            prefix,
            params,
            config.max_boost_rounds,
            config.early_stopping_rounds,
            verbose=False,
        )
        record = {
            "trial": trial.number,
            **trial.params,
            **metrics,
        }
        trial_records.append(record)
        del booster
        gc.collect()
        return float(metrics["mean_rank_ic"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.seed),
    )
    study.optimize(objective, n_trials=config.tuning_trials, show_progress_bar=False)
    table = pd.DataFrame(trial_records).sort_values(
        ["mean_rank_ic", "std_rank_ic"],
        ascending=[False, True],
    )
    top_params = []
    for row in table.head(config.verify_top_parameter_sets).to_dict(orient="records"):
        params = base_lgb_params()
        for key in (
            "learning_rate",
            "num_leaves",
            "min_data_in_leaf",
            "feature_fraction",
            "bagging_fraction",
            "lambda_l1",
            "lambda_l2",
            "max_bin",
        ):
            params[key] = row[key]
        top_params.append(params)
    return table, top_params


def verify_parameter_sets(
    data: CompetitionData,
    history_cache: dict[str, np.memmap],
    selected_features: np.ndarray,
    history_features: np.ndarray,
    interaction_specs: Sequence[dict],
    imputation_mode: str,
    stage: str,
    parameter_sets: Sequence[dict],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for fold in FOLDS:
        state = fit_category_state(
            data,
            fold.train_start,
            fold.train_stop,
            history_features,
            config,
        )
        train_matrix = build_feature_matrix(
            data,
            history_cache,
            selected_features,
            history_features,
            interaction_specs,
            state,
            fold.train_start,
            fold.train_stop,
            config.tuning_stock_cap,
            "train",
            imputation_mode,
            f"verify_{fold.name}_train",
            config,
        )
        valid_matrix = build_feature_matrix(
            data,
            history_cache,
            selected_features,
            history_features,
            interaction_specs,
            state,
            fold.valid_start,
            fold.valid_stop,
            config.tuning_stock_cap,
            "valid",
            imputation_mode,
            f"verify_{fold.name}_valid",
            config,
        )
        prefix = train_matrix.block_ends[stage]
        for parameter_index, params in enumerate(parameter_sets):
            booster, metrics = train_ranker(
                train_matrix,
                valid_matrix,
                prefix,
                params,
                config.max_boost_rounds,
                config.early_stopping_rounds,
                verbose=False,
            )
            rows.append(
                {
                    "parameter_set": parameter_index,
                    "fold": fold.name,
                    **metrics,
                }
            )
            del booster
            gc.collect()
        remove_feature_matrix(train_matrix)
        remove_feature_matrix(valid_matrix)

    table = pd.DataFrame(rows)
    summary = (
        table.groupby("parameter_set", as_index=False)
        .agg(
            mean_fold_rank_ic=("mean_rank_ic", "mean"),
            median_fold_rank_ic=("mean_rank_ic", "median"),
            std_across_folds=("mean_rank_ic", "std"),
            mean_best_iteration=("best_iteration", "mean"),
        )
        .sort_values(
            ["mean_fold_rank_ic", "std_across_folds"],
            ascending=[False, True],
        )
    )
    winner_index = int(summary.iloc[0]["parameter_set"])
    return table.merge(summary, on="parameter_set"), dict(parameter_sets[winner_index])


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
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


def _package_versions() -> dict[str, str]:
    import lightgbm
    import matplotlib
    import nbclient
    import nbformat

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "optuna": optuna.__version__,
        "zstandard": zstd.__version__,
        "matplotlib": matplotlib.__version__,
        "nbformat": nbformat.__version__,
        "nbclient": nbclient.__version__,
    }


def _format_markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


def _write_experiment_report_legacy(
    output_path: Path,
    numeric_summary: pd.DataFrame,
    interaction_table: pd.DataFrame,
    reconstruction_table: pd.DataFrame,
    imputation_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    tuning_table: pd.DataFrame,
    verification_table: pd.DataFrame,
    official_metrics: dict,
    selected_features: np.ndarray,
    selected_interactions: Sequence[dict],
    selected_imputation: str,
    selected_stage: str,
    winner: str,
    prediction_summary: dict,
) -> None:
    top_numeric = numeric_summary.sort_values(
        "median_abs_fold_ic", ascending=False
    )[
        [
            "feature_index",
            "feature",
            "median_abs_fold_ic",
            "fold_sign_stability",
            "correlation_cluster",
            "selected_top40",
        ]
    ]
    ablation_summary = (
        ablation_table.groupby("stage", as_index=False)
        .agg(
            mean_rank_ic=("mean_rank_ic", "mean"),
            median_rank_ic=("mean_rank_ic", "median"),
            std_rank_ic=("mean_rank_ic", "std"),
        )
        .sort_values("median_rank_ic", ascending=False)
    )
    verification_summary = (
        verification_table.groupby("parameter_set", as_index=False)
        .agg(
            mean_rank_ic=("mean_rank_ic", "mean"),
            median_rank_ic=("mean_rank_ic", "median"),
            std_across_folds=("mean_rank_ic", "std"),
        )
        .sort_values("mean_rank_ic", ascending=False)
    )
    text = f"""# Y1 特征工程与排序模型实验报告

## 结论

- 线性基线验证 RankIC：{LINEAR_BASELINE_IC:.6f}
- TCN 基线验证 RankIC：{TCN_BASELINE_IC:.6f}
- 新 LightGBM 官方验证 RankIC：{official_metrics['mean_rank_ic']:.6f}
- 最终获胜单模型：`{winner}`
- 采用特征阶段：`{selected_stage}`
- 采用缺失方案：`{selected_imputation}`
- 最终选择数值特征：{len(selected_features)} 个
- 最终显式交叉项：{len(selected_interactions)} 个

## 方法与数据口径

- 目标仅为 Y1。
- 监督样本严格使用 `mask_x & mask_y & finite(y1)`。
- 采用三个扩展窗口走步折；官方验证区间仅用于最终晋级判断。
- Y1 不插值；测试集 `mask_y=True` 位置预测，其他位置填写 0.5。
- 当前 `mask_x=False` 是进入股票池前的长前导区，不伪造当前训练样本。

## 数值特征筛选

{_format_markdown_table(top_numeric, 25)}

## 交叉项残差信号

{_format_markdown_table(interaction_table, 20)}

## 缺失处理

### 人工遮挡重建

{_format_markdown_table(reconstruction_table)}

### 下游 RankIC 对照

{_format_markdown_table(imputation_table)}

## 类别与特征组消融

{_format_markdown_table(ablation_summary)}

## 参数搜索

{_format_markdown_table(tuning_table, 10)}

## 三折复核

{_format_markdown_table(verification_summary)}

## 官方验证

```json
{json.dumps(_json_ready(official_metrics), ensure_ascii=False, indent=2)}
```

## 最终预测文件检查

```json
{json.dumps(_json_ready(prediction_summary), ensure_ascii=False, indent=2)}
```

## 解释与限制

- LambdaRank 使用 64 级线性相关度近似连续 Y1 排序，并以逐时间 RankIC 作为早停指标。
- 类别编码、频率、历史目标均值和分位数截断只在对应训练区间拟合。
- 显式交叉项和趋势填充只有在走步验证达到晋级阈值时才保留。
- 若新模型未超过 TCN，提交文件保持使用验证更优的 TCN 单模型；LightGBM 结果仍保留用于复核。
"""
    output_path.write_text(text, encoding="utf-8")


def write_experiment_report(
    output_path: Path,
    numeric_summary: pd.DataFrame,
    interaction_table: pd.DataFrame,
    reconstruction_table: pd.DataFrame,
    imputation_table: pd.DataFrame,
    ablation_table: pd.DataFrame,
    tuning_table: pd.DataFrame,
    verification_table: pd.DataFrame,
    official_metrics: dict,
    selected_features: np.ndarray,
    selected_interactions: Sequence[dict],
    selected_imputation: str,
    selected_stage: str,
    winner: str,
    prediction_summary: dict,
) -> None:
    top_numeric = numeric_summary.sort_values(
        "median_abs_fold_ic", ascending=False
    )[
        [
            "feature_index",
            "feature",
            "median_abs_fold_ic",
            "fold_sign_stability",
            "correlation_cluster",
            "selected_top40",
        ]
    ]
    ablation_summary = (
        ablation_table.groupby("stage", as_index=False)
        .agg(
            mean_rank_ic=("mean_rank_ic", "mean"),
            median_rank_ic=("mean_rank_ic", "median"),
            std_rank_ic=("mean_rank_ic", "std"),
        )
        .sort_values("median_rank_ic", ascending=False)
    )
    verification_summary = (
        verification_table.groupby("parameter_set", as_index=False)
        .agg(
            mean_rank_ic=("mean_rank_ic", "mean"),
            median_rank_ic=("mean_rank_ic", "median"),
            std_across_folds=("mean_rank_ic", "std"),
        )
        .sort_values("mean_rank_ic", ascending=False)
    )
    text = f"""# Y1 特征工程与排序模型实验报告

## 结论

- 线性基线验证 RankIC：{LINEAR_BASELINE_IC:.6f}
- TCN 基线验证 RankIC：{TCN_BASELINE_IC:.6f}
- 新 LightGBM 官方验证 RankIC：{official_metrics['mean_rank_ic']:.6f}
- 最终获胜单模型：`{winner}`
- 采用特征阶段：`{selected_stage}`
- 采用缺失方案：`{selected_imputation}`
- 最终选择数值特征：{len(selected_features)} 个
- 最终显式交叉项：{len(selected_interactions)} 个

## 方法与数据口径

- 目标仅为 Y1。
- 监督样本严格使用 `mask_x & mask_y & finite(y1)`。
- 采用三个扩展窗口走步折；官方验证区间仅用于最终晋级判断。
- Y1 不插值；测试集 `mask_y=True` 位置预测，其余位置填充 0.5。
- 当前 `mask_x=False` 是进入股票池前的长前导区，不伪造当前训练样本。

## 数值特征筛选

{_format_markdown_table(top_numeric, 25)}

## 交叉项残差信号

{_format_markdown_table(interaction_table, 20)}

## 缺失处理

### 人工遮挡重建

{_format_markdown_table(reconstruction_table)}

### 下游 RankIC 对照

{_format_markdown_table(imputation_table)}

## 类别与特征组消融

{_format_markdown_table(ablation_summary)}

## 参数搜索

{_format_markdown_table(tuning_table, 10)}

## 三折复核

{_format_markdown_table(verification_summary)}

## 官方验证

```json
{json.dumps(_json_ready(official_metrics), ensure_ascii=False, indent=2)}
```

## 最终预测文件检查

```json
{json.dumps(_json_ready(prediction_summary), ensure_ascii=False, indent=2)}
```

## 解释与限制

- LambdaRank 使用 64 级线性相关度近似连续 Y1 排序，并以逐时间 RankIC 作为早停指标。
- 类别映射、频率、历史目标均值和分位数截断只在对应训练区间拟合。
- 显式交叉项和趋势填充只有在走步验证达到晋级阈值时才保留。
- 若新模型未超过 TCN，提交文件保持使用验证更优的 TCN 单模型；LightGBM 结果仍保留用于复核。
"""
    output_path.write_text(text, encoding="utf-8")


def run_pipeline(config: PipelineConfig | None = None) -> dict:
    config = config or PipelineConfig()
    config.output_dir_obj.mkdir(parents=True, exist_ok=True)
    config.cache_dir_obj.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    log("Loading competition data through read-only memmaps", config.verbose)
    data = CompetitionData(config)

    log("Discovering stable Y1 numeric features", config.verbose)
    numeric_summary, selected_features, mean_correlation = discover_numeric_features(
        data, config
    )
    history_features = selected_features[: config.history_feature_count]
    numeric_summary.to_csv(
        config.output_dir_obj / "numeric_feature_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    history_paths = build_history_cache(data, history_features, config)
    history_cache = open_history_cache(history_paths, history_features.size)

    log("Screening explicit interaction candidates against residual Y1", config.verbose)
    interaction_specs, interaction_table = screen_interactions(
        data,
        numeric_summary,
        selected_features,
        mean_correlation,
        config,
    )
    interaction_table.to_csv(
        config.output_dir_obj / "interaction_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_state = fit_category_state(
        data,
        TRAIN_START,
        VALID_START,
        history_features,
        config,
    )
    reconstruction_table = imputation_reconstruction_diagnostic(
        data,
        history_features,
        base_state,
        config,
    )
    log("Comparing native missing handling with causal EWMA trend filling", config.verbose)
    imputation_table, selected_imputation = compare_imputation_modes(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        config,
    )

    log("Running three-fold cumulative feature-stage ablations", config.verbose)
    ablation_table, selected_stage = run_feature_stage_ablation(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        selected_imputation,
        config,
    )
    ablation_table.to_csv(
        config.output_dir_obj / "feature_stage_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    tuning_fold = FOLDS[-1]
    tuning_state = fit_category_state(
        data,
        tuning_fold.train_start,
        tuning_fold.train_stop,
        history_features,
        config,
    )
    log("Building bounded matrices for the 30-trial parameter search", config.verbose)
    tuning_train = build_feature_matrix(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        tuning_state,
        tuning_fold.train_start,
        tuning_fold.train_stop,
        config.tuning_stock_cap,
        "train",
        selected_imputation,
        "tuning_train",
        config,
    )
    tuning_valid = build_feature_matrix(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        tuning_state,
        tuning_fold.valid_start,
        tuning_fold.valid_stop,
        config.tuning_stock_cap,
        "valid",
        selected_imputation,
        "tuning_valid",
        config,
    )
    log("Running Optuna parameter search", config.verbose)
    tuning_table, parameter_sets = tune_parameters(
        tuning_train,
        tuning_valid,
        selected_stage,
        config,
    )
    tuning_table.to_csv(
        config.output_dir_obj / "parameter_search.csv",
        index=False,
        encoding="utf-8-sig",
    )
    remove_feature_matrix(tuning_train)
    remove_feature_matrix(tuning_valid)

    log("Verifying the top five parameter sets on all walk-forward folds", config.verbose)
    verification_table, winning_params = verify_parameter_sets(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        selected_imputation,
        selected_stage,
        parameter_sets,
        config,
    )
    verification_table.to_csv(
        config.output_dir_obj / "parameter_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("Training on the official training interval and evaluating official validation", config.verbose)
    official_state = fit_category_state(
        data,
        TRAIN_START,
        VALID_START,
        history_features,
        config,
    )
    official_train = build_feature_matrix(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        official_state,
        TRAIN_START,
        VALID_START,
        config.final_train_stock_cap,
        "train",
        selected_imputation,
        "official_train",
        config,
    )
    official_valid = build_feature_matrix(
        data,
        history_cache,
        selected_features,
        history_features,
        interaction_specs,
        official_state,
        VALID_START,
        TEST_START,
        None,
        "valid",
        selected_imputation,
        "official_valid",
        config,
    )
    prefix = official_train.block_ends[selected_stage]
    official_booster, official_metrics = train_ranker(
        official_train,
        official_valid,
        prefix,
        winning_params,
        config.max_boost_rounds,
        config.early_stopping_rounds,
        verbose=True,
    )
    save_booster(official_booster, config.output_dir_obj / "model.txt")
    official_best_iteration = max(1, int(official_metrics["best_iteration"]))
    remove_feature_matrix(official_train)
    remove_feature_matrix(official_valid)
    del official_booster
    gc.collect()

    new_model_wins = official_metrics["mean_rank_ic"] > TCN_BASELINE_IC
    tcn_prediction_path = Path("../y1_tcn_baseline/outputs/tcn_y1.npy").resolve()
    winner = "lightgbm_lambdarank" if new_model_wins else "tcn_y1"

    if new_model_wins:
        log("LightGBM passed the TCN promotion gate; retraining through official validation", config.verbose)
        final_state = fit_category_state(
            data,
            TRAIN_START,
            TEST_START,
            history_features,
            config,
        )
        final_train = build_feature_matrix(
            data,
            history_cache,
            selected_features,
            history_features,
            interaction_specs,
            final_state,
            TRAIN_START,
            TEST_START,
            config.final_train_stock_cap,
            "train",
            selected_imputation,
            "final_train",
            config,
        )
        final_test = build_feature_matrix(
            data,
            history_cache,
            selected_features,
            history_features,
            interaction_specs,
            final_state,
            TEST_START,
            T,
            None,
            "test",
            selected_imputation,
            "final_test",
            config,
        )
        final_model = train_ranker_no_validation(
            final_train,
            final_train.block_ends[selected_stage],
            winning_params,
            official_best_iteration,
        )
        save_booster(final_model, config.output_dir_obj / "model.txt")
        test_values = final_test.open()
        raw_predictions = final_model.predict(
            test_values[:, : final_test.block_ends[selected_stage]],
            num_iteration=official_best_iteration,
        )
        predictions = np.full((T - TEST_START, S), 0.5, dtype=np.float32)
        offset = 0
        for output_index, time_idx in enumerate(range(TEST_START, T)):
            stocks = data.eligible_stocks(time_idx, require_label=False)
            group_size = stocks.size
            scores = raw_predictions[offset : offset + group_size]
            ranks = rankdata(scores, method="average")
            percentile = (
                (ranks - 1.0) / max(group_size - 1, 1)
            ).astype(np.float32)
            predictions[output_index, stocks] = percentile
            offset += group_size
        del test_values, raw_predictions, final_model
        remove_feature_matrix(final_train)
        remove_feature_matrix(final_test)
    else:
        if not tcn_prediction_path.exists():
            raise FileNotFoundError(
                "LightGBM did not beat TCN and ../y1_tcn_baseline/outputs/tcn_y1.npy is missing"
            )
        predictions = np.load(tcn_prediction_path).astype(np.float32, copy=True)

    prediction_path = config.output_dir_obj / "y1_best.npy"
    np.save(prediction_path, predictions)
    loaded = np.load(prediction_path)
    assert loaded.shape == (T - TEST_START, S)
    assert loaded.dtype == np.float32
    assert np.all(np.isfinite(loaded))
    prediction_summary = {
        "path": str(prediction_path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "minimum": float(loaded.min()),
        "maximum": float(loaded.max()),
        "mean": float(loaded.mean()),
        "neutral_count": int(np.count_nonzero(loaded == 0.5)),
        "file_size_mb": float(prediction_path.stat().st_size / 1024**2),
        "np_load_verified": True,
    }
    all_feature_names, _, _ = feature_layout(
        selected_features,
        history_features,
        interaction_specs,
    )
    final_interactions = (
        list(interaction_specs) if selected_stage == "interaction" else []
    )

    manifest = {
        "target": "y1",
        "winner": winner,
        "baselines": {
            "linear_mean_rank_ic": LINEAR_BASELINE_IC,
            "tcn_mean_rank_ic": TCN_BASELINE_IC,
        },
        "official_validation": official_metrics,
        "selected_stage": selected_stage,
        "selected_imputation": selected_imputation,
        "selected_numeric_features": selected_features,
        "selected_history_features": history_features,
        "selected_interactions": interaction_specs,
        "screened_interaction_candidates": interaction_specs,
        "final_selected_interactions": final_interactions,
        "final_feature_names": all_feature_names[:prefix],
        "winning_lightgbm_params": winning_params,
        "feature_prefix": int(prefix),
        "category_policy": {
            "low_cardinality": "LightGBM native categorical with train-only mapping",
            "cat_5": "native category + frequency + log frequency + past-only smoothed target mean",
            "unknown": "dedicated unknown bucket",
            "group_features": ["cat_1", "cat_6"],
        },
        "category_mappings": {
            "included_in_final_model": selected_stage != "numeric",
            "status": (
                "not_applicable_numeric_stage_won"
                if selected_stage == "numeric"
                else "fit_on_final_training_interval"
            ),
            "unknown_bucket_policy": "dedicated unknown code per category",
        },
        "walk_forward_folds": [asdict(fold) for fold in FOLDS],
        "versions": _package_versions(),
        "config": asdict(config),
        "prediction": prediction_summary,
        "runtime_seconds": time.time() - started_at,
    }
    manifest_path = config.output_dir_obj / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_experiment_report(
        config.output_dir_obj / "y1_experiment_report.md",
        numeric_summary,
        interaction_table,
        reconstruction_table,
        imputation_table,
        ablation_table,
        tuning_table,
        verification_table,
        official_metrics,
        selected_features,
        final_interactions,
        selected_imputation,
        selected_stage,
        winner,
        prediction_summary,
    )
    small_result = {
        "winner": winner,
        "official_validation": official_metrics,
        "selected_stage": selected_stage,
        "selected_imputation": selected_imputation,
        "selected_numeric_features": selected_features.tolist(),
        "selected_interaction_count": len(final_interactions),
        "prediction": prediction_summary,
        "runtime_seconds": time.time() - started_at,
        "output_dir": str(config.output_dir_obj),
    }
    return small_result
