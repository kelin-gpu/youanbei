#!/usr/bin/env python3
"""面向动态股票池的因果多模型排序预测系统：Y1 数据分析执行脚本。

对项目输入数据 data.z（等价缓存 payload.pkl）执行五个维度的数据分析，
并把全部数值结果写入 01_analysis/outputs/：

  D1 掩码覆盖与股票池       -> mask_coverage.csv, coverage_by_time 数组
  D2 标签分布与稳定性       -> label_profile.csv, label_time_series.csv
  D3 数值特征质量与漂移     -> numeric_feature_profile.csv
  D4 单特征 RankIC          -> rankic_summary.csv
  D5 类别特征与编码建议     -> category_profile.csv, category_change.csv

最终汇总为 analysis_results.json（HTML 报告的唯一致数值来源）与
manifest.json（运行元数据）。支持 --seed 覆盖与 --selftest 自检。

统计口径与 01_analysis/data_analysis.ipynb 完全一致：
- 掩码覆盖按四个官方阶段统计；占位验证扫描 mask_x=False 的数值。
- 标签统计只在 usable = mask_x & mask_y & finite(y1) 上计算。
- 数值特征统计统一使用 mask_x & mask_y；分位数/PSI 阈值只由训练期拟合。
- 类别特征 pretrain 用 mask_x，其余阶段用 mask_x & mask_y。

运行（固定已核验环境）：
  D:\\anaconda\\anaconda_data\\envs\\jingge_ts\\python.exe run_data_analysis.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data.z"
OUTPUT_DIR = PROJECT_ROOT / "01_analysis" / "outputs"
PAYLOAD_CACHE = (
    PROJECT_ROOT
    / "archive"
    / "legacy_structure"
    / "03_model_training"
    / "y1_pipeline_v2"
    / ".cache_dp_v2"
    / "payload.pkl"
)

ANALYSIS_SEED = 20260724
TIME_CHUNK_SIZE = 16
MAX_SAMPLE_TIME_POINTS = 300
SAMPLE_STOCKS_PER_TIME = 300
TRAIN_IC_TIME_POINTS = 80
TOP_N = 15
LABEL_QUANTILES = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]

SPLIT_NAMES = ["pretrain", "train", "valid", "test"]

# phase1_diagnostic_report.txt 提供的覆盖率锚点（容差 1e-4）。
PHASE1_COVERAGE_ANCHORS = {
    "pretrain": {"mask_x_true_rate": 0.337495, "mask_y_true_rate": 0.281950},
    "train": {"mask_x_true_rate": 0.572496, "mask_y_true_rate": 0.505152, "usable_y1_rate": 0.505152},
    "valid": {"mask_x_true_rate": 0.848913, "mask_y_true_rate": 0.765837, "usable_y1_rate": 0.765837},
    "test": {"mask_x_true_rate": 0.950534, "mask_y_true_rate": 0.874882},
}
PHASE1_SEMANTIC_MISSING_ANCHOR = 0.394184
PHASE1_TARGET_MISSING_ANCHOR = 0.607375


def to_float(value):
    """把 numpy 标量转换为 python float，便于 JSON 序列化。"""
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def load_data() -> tuple[dict, str, float]:
    """加载数据：优先复用 payload.pkl 缓存，否则解压 data.z。"""
    if PAYLOAD_CACHE.exists():
        started = time.perf_counter()
        with PAYLOAD_CACHE.open("rb") as handle:
            data = pickle.load(handle)
        elapsed = time.perf_counter() - started
        return data, str(PAYLOAD_CACHE), elapsed
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError("缺少 zstandard 且没有 payload.pkl 缓存。") from exc
    started = time.perf_counter()
    with DATA_PATH.open("rb") as handle:
        compressed = pickle.load(handle)
    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    data = pickle.loads(decompressed)
    elapsed = time.perf_counter() - started
    return data, str(DATA_PATH), elapsed


def validate_data(data: dict) -> tuple[int, int, int, int]:
    T, stock_count, feature_count = data["num_x"].shape
    category_count = data["cat_x"].shape[2]
    train_start = int(data["train_start_idx"])
    valid_start = int(data["valid_start_idx"])
    test_start = int(data["test_start_idx"])
    assert 0 <= train_start <= valid_start <= test_start <= T
    assert data["cat_x"].shape[:2] == (T, stock_count)
    assert data["y1"].shape == (T, stock_count)
    assert data["mask_x"].shape == (T, stock_count)
    assert data["mask_y"].shape == (T, stock_count)
    return train_start, valid_start, test_start, feature_count


def make_split_slices(train_start, valid_start, test_start, T):
    return {
        "pretrain": slice(0, train_start),
        "train": slice(train_start, valid_start),
        "valid": slice(valid_start, test_start),
        "test": slice(test_start, T),
    }


# ---------------------------------------------------------------- D1
def analyze_coverage(data, split_slices, T, stock_count):
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    y1 = data["y1"]

    mask_x_count = mask_x.sum(axis=1, dtype=np.int64)
    mask_y_count = mask_y.sum(axis=1, dtype=np.int64)
    finite_y1_count = np.isfinite(y1).sum(axis=1, dtype=np.int64)
    usable_y1_count = (mask_x & mask_y & np.isfinite(y1)).sum(axis=1, dtype=np.int64)

    by_split = []
    for split_name in SPLIT_NAMES:
        start, stop = split_slices[split_name].start, split_slices[split_name].stop
        total_positions = (stop - start) * stock_count
        by_split.append(
            {
                "split": split_name,
                "time_count": stop - start,
                "mask_x_true_rate": float(mask_x_count[start:stop].sum() / total_positions),
                "mask_x_per_time_median": float(np.median(mask_x_count[start:stop])),
                "mask_y_true_rate": float(mask_y_count[start:stop].sum() / total_positions),
                "finite_y1_rate": float(finite_y1_count[start:stop].sum() / total_positions),
                "usable_y1_rate": float(usable_y1_count[start:stop].sum() / total_positions),
            }
        )

    placeholder = []
    num_x = data["num_x"]
    for split_name in SPLIT_NAMES:
        split_slice = split_slices[split_name]
        invalid_rows = 0
        invalid_zero_elements = 0
        invalid_elements = 0
        invalid_allzero_rows = 0
        invalid_min = np.inf
        invalid_max = -np.inf
        for chunk_start in range(split_slice.start, split_slice.stop, TIME_CHUNK_SIZE):
            chunk_stop = min(chunk_start + TIME_CHUNK_SIZE, split_slice.stop)
            invalid_mask = ~mask_x[chunk_start:chunk_stop]
            if not np.any(invalid_mask):
                continue
            values = num_x[chunk_start:chunk_stop][invalid_mask]
            invalid_rows += values.shape[0]
            invalid_elements += values.size
            invalid_zero_elements += np.count_nonzero(values == 0)
            invalid_allzero_rows += np.count_nonzero(np.all(values == 0, axis=1))
            invalid_min = min(invalid_min, float(values.min()))
            invalid_max = max(invalid_max, float(values.max()))
        placeholder.append(
            {
                "split": split_name,
                "invalid_rows": int(invalid_rows),
                "numeric_zero_rate": float(invalid_zero_elements / max(invalid_elements, 1)),
                "allzero_row_rate": float(invalid_allzero_rows / max(invalid_rows, 1)),
                "invalid_numeric_min": to_float(invalid_min),
                "invalid_numeric_max": to_float(invalid_max),
            }
        )
        assert np.allclose(invalid_zero_elements / max(invalid_elements, 1), 1.0)
        assert np.allclose(invalid_allzero_rows / max(invalid_rows, 1), 1.0)

    by_time = {
        "time": np.arange(T, dtype=np.int64).tolist(),
        "mask_x_count": mask_x_count.tolist(),
        "mask_y_count": mask_y_count.tolist(),
        "finite_y1_count": finite_y1_count.tolist(),
        "usable_y1_count": usable_y1_count.tolist(),
    }
    return {"by_split": by_split, "placeholder": placeholder, "by_time": by_time}


# ---------------------------------------------------------------- D2
def analyze_label(data, split_slices, mask_x, mask_y, rng):
    y1 = data["y1"]
    profile_rows = []
    label_samples = {}

    for split_name in ("train", "valid"):
        split_slice = split_slices[split_name]
        usable_mask = mask_x[split_slice] & mask_y[split_slice] & np.isfinite(y1[split_slice])
        values = y1[split_slice][usable_mask].astype(np.float64, copy=False)
        quantiles = np.quantile(values, LABEL_QUANTILES)
        q1, median, q3 = quantiles[4], quantiles[5], quantiles[6]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        profile_rows.append(
            {
                "split": split_name,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(quantiles[0]),
                "p001": float(quantiles[1]),
                "p01": float(quantiles[2]),
                "p05": float(quantiles[3]),
                "p25": float(q1),
                "median": float(median),
                "p75": float(q3),
                "p95": float(quantiles[7]),
                "p99": float(quantiles[8]),
                "p999": float(quantiles[9]),
                "max": float(quantiles[10]),
                "iqr_outlier_rate": float(np.mean((values < lower) | (values > upper))),
            }
        )
        sample_size = min(values.size, 200_000)
        sampled_indices = rng.choice(values.size, size=sample_size, replace=False)
        label_samples[split_name] = values[sampled_indices]

    label_time_rows = []
    for time_idx in range(split_slices["train"].start, split_slices["test"].stop):
        usable_mask = mask_x[time_idx] & mask_y[time_idx] & np.isfinite(y1[time_idx])
        values = y1[time_idx, usable_mask]
        if values.size:
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            label_time_rows.append(
                {
                    "time": int(time_idx),
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "q1": float(q1),
                    "median": float(median),
                    "q3": float(q3),
                }
            )

    histogram = {}
    bin_edges = np.linspace(0.0, 1.0, 51)
    for split_name in ("train", "valid"):
        counts, _ = np.histogram(label_samples[split_name], bins=bin_edges)
        histogram[split_name] = {
            "edges": bin_edges.tolist(),
            "counts": counts.tolist(),
        }

    time_series = {
        "time": [row["time"] for row in label_time_rows],
        "n": [row["n"] for row in label_time_rows],
        "mean": [row["mean"] for row in label_time_rows],
        "std": [row["std"] for row in label_time_rows],
    }
    return {
        "profile": profile_rows,
        "histogram": histogram,
        "time_series": time_series,
        "time_rows": label_time_rows,
    }


# ---------------------------------------------------------------- D3
def profile_numeric_split(split_name, split_slice, data, mask_x, mask_y, rng, feature_count):
    num_x = data["num_x"]
    count = 0
    sums = np.zeros(feature_count, dtype=np.float64)
    squared_sums = np.zeros(feature_count, dtype=np.float64)
    zero_counts = np.zeros(feature_count, dtype=np.int64)
    nonfinite_counts = np.zeros(feature_count, dtype=np.int64)
    minimum = np.full(feature_count, np.inf)
    maximum = np.full(feature_count, -np.inf)

    for chunk_start in range(split_slice.start, split_slice.stop, TIME_CHUNK_SIZE):
        chunk_stop = min(chunk_start + TIME_CHUNK_SIZE, split_slice.stop)
        selected_mask = mask_x[chunk_start:chunk_stop] & mask_y[chunk_start:chunk_stop]
        values = num_x[chunk_start:chunk_stop][selected_mask]
        if values.size == 0:
            continue
        count += values.shape[0]
        sums += values.sum(axis=0, dtype=np.float64)
        squared_sums += np.einsum("ij,ij->j", values, values, dtype=np.float64)
        zero_counts += np.count_nonzero(values == 0, axis=0)
        nonfinite_counts += np.count_nonzero(~np.isfinite(values), axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))

    mean = sums / max(count, 1)
    std = np.sqrt(np.maximum(squared_sums / max(count, 1) - mean**2, 0))

    sampled_rows = []
    sampled_times = np.linspace(
        split_slice.start,
        split_slice.stop - 1,
        min(split_slice.stop - split_slice.start, MAX_SAMPLE_TIME_POINTS),
        dtype=int,
    )
    for time_idx in sampled_times:
        available_stocks = np.flatnonzero(mask_x[time_idx] & mask_y[time_idx])
        if available_stocks.size == 0:
            continue
        sample_size = min(SAMPLE_STOCKS_PER_TIME, available_stocks.size)
        chosen_stocks = rng.choice(available_stocks, size=sample_size, replace=False)
        sampled_rows.append(num_x[time_idx, chosen_stocks])

    sample = np.concatenate(sampled_rows, axis=0) if sampled_rows else np.zeros((0, feature_count), dtype=np.float32)
    quantiles = np.quantile(sample, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)

    return {
        "split": split_name,
        "n": count,
        "mean": mean,
        "std": std,
        "zero_rate": zero_counts / max(count, 1),
        "nonfinite_rate": nonfinite_counts / max(count, 1),
        "min": minimum,
        "max": maximum,
        "p01": quantiles[0],
        "p25": quantiles[1],
        "median": quantiles[2],
        "p75": quantiles[3],
        "p99": quantiles[4],
        "sample": sample,
    }


def calculate_psi(train_values, comparison_values, feature_index):
    bin_edges = np.unique(np.quantile(train_values[:, feature_index], np.linspace(0, 1, 11)))
    if bin_edges.size < 3:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    train_counts = np.histogram(train_values[:, feature_index], bins=bin_edges)[0].astype(np.float64)
    comparison_counts = np.histogram(comparison_values[:, feature_index], bins=bin_edges)[0].astype(np.float64)
    train_share = (train_counts + 0.5) / (train_counts.sum() + 0.5 * train_counts.size)
    comparison_share = (comparison_counts + 0.5) / (
        comparison_counts.sum() + 0.5 * comparison_counts.size
    )
    return float(
        np.sum((comparison_share - train_share) * np.log(comparison_share / train_share))
    )


def analyze_numeric(data, split_slices, mask_x, mask_y, rng, feature_count):
    profiles = {
        split_name: profile_numeric_split(
            split_name, split_slices[split_name], data, mask_x, mask_y, rng, feature_count
        )
        for split_name in ("train", "valid", "test")
    }
    sampling_summary = [
        {"split": split_name, "rows": int(profiles[split_name]["n"]),
         "sample_rows": int(profiles[split_name]["sample"].shape[0])}
        for split_name in ("train", "valid", "test")
    ]

    train_profile = profiles["train"]
    train_iqr = train_profile["p75"] - train_profile["p25"]
    outlier_lower = train_profile["p25"] - 1.5 * train_iqr
    outlier_upper = train_profile["p75"] + 1.5 * train_iqr

    outlier_rates = {
        split_name: np.mean(
            (profiles[split_name]["sample"] < outlier_lower)
            | (profiles[split_name]["sample"] > outlier_upper),
            axis=0,
        )
        for split_name in ("train", "valid", "test")
    }

    valid_psi = np.asarray(
        [
            calculate_psi(train_profile["sample"], profiles["valid"]["sample"], feature_index)
            for feature_index in range(feature_count)
        ]
    )
    test_psi = np.asarray(
        [
            calculate_psi(train_profile["sample"], profiles["test"]["sample"], feature_index)
            for feature_index in range(feature_count)
        ]
    )

    feature_rows = []
    for feature_index in range(feature_count):
        valid_shift = (
            profiles["valid"]["mean"][feature_index] - train_profile["mean"][feature_index]
        ) / max(train_profile["std"][feature_index], 1e-12)
        test_shift = (
            profiles["test"]["mean"][feature_index] - train_profile["mean"][feature_index]
        ) / max(train_profile["std"][feature_index], 1e-12)
        feature_rows.append(
            {
                "feature": f"num_{feature_index}",
                "feature_index": feature_index,
                "train_mean": float(train_profile["mean"][feature_index]),
                "train_std": float(train_profile["std"][feature_index]),
                "nonfinite_rate": float(train_profile["nonfinite_rate"][feature_index]),
                "zero_rate": float(train_profile["zero_rate"][feature_index]),
                "train_min": float(train_profile["min"][feature_index]),
                "train_p01": float(train_profile["p01"][feature_index]),
                "train_median": float(train_profile["median"][feature_index]),
                "train_p99": float(train_profile["p99"][feature_index]),
                "train_max": float(train_profile["max"][feature_index]),
                "max_abs": float(
                    max(abs(train_profile["min"][feature_index]), abs(train_profile["max"][feature_index]))
                ),
                "train_outlier_rate": float(outlier_rates["train"][feature_index]),
                "valid_outlier_rate": float(outlier_rates["valid"][feature_index]),
                "test_outlier_rate": float(outlier_rates["test"][feature_index]),
                "valid_mean_shift_sd": float(valid_shift),
                "test_mean_shift_sd": float(test_shift),
                "valid_psi": float(valid_psi[feature_index]),
                "test_psi": float(test_psi[feature_index]),
            }
        )
    for row in feature_rows:
        row["max_abs_mean_shift"] = float(
            max(abs(row["valid_mean_shift_sd"]), abs(row["test_mean_shift_sd"]))
        )
        row["max_psi"] = float(max(row["valid_psi"], row["test_psi"]))

    nonfinite_max = max(row["nonfinite_rate"] for row in feature_rows)
    assert nonfinite_max == 0
    assert all(row["train_std"] > 0 for row in feature_rows)

    return {
        "sampling_summary": sampling_summary,
        "features": feature_rows,
    }


# ---------------------------------------------------------------- D4
def rank_columns(values):
    return pd.DataFrame(values).rank(method="average", axis=0).to_numpy(dtype=np.float64, copy=True)


def calculate_univariate_rank_ic(data, mask_x, mask_y, target, time_indices, feature_count):
    time_ic_rows = []
    for time_idx in time_indices:
        usable_mask = mask_x[time_idx] & mask_y[time_idx] & np.isfinite(target[time_idx])
        if np.count_nonzero(usable_mask) < 20:
            continue
        features = data["num_x"][time_idx, usable_mask]
        labels = target[time_idx, usable_mask]
        feature_ranks = rank_columns(features)
        label_ranks = pd.Series(labels).rank(method="average").to_numpy(dtype=np.float64, copy=True)
        feature_ranks -= feature_ranks.mean(axis=0)
        label_ranks -= label_ranks.mean()
        denominator = np.sqrt(
            np.einsum("ij,ij->j", feature_ranks, feature_ranks) * np.dot(label_ranks, label_ranks)
        )
        correlations = np.divide(
            feature_ranks.T @ label_ranks,
            denominator,
            out=np.full(feature_count, np.nan),
            where=denominator > 0,
        )
        time_ic_rows.append(correlations)
    return np.vstack(time_ic_rows) if time_ic_rows else np.zeros((0, feature_count))


def analyze_rank_ic(data, split_slices, mask_x, mask_y, feature_count):
    train_ic_times = np.linspace(
        split_slices["train"].start,
        split_slices["train"].stop - 1,
        TRAIN_IC_TIME_POINTS,
        dtype=int,
    )
    valid_ic_times = np.arange(split_slices["valid"].start, split_slices["valid"].stop)

    summary_rows = []
    for split_name, time_indices in (("train_sample", train_ic_times), ("valid", valid_ic_times)):
        ic_matrix = calculate_univariate_rank_ic(
            data, mask_x, mask_y, data["y1"], time_indices, feature_count
        )
        finite_ic = np.isfinite(ic_matrix)
        valid_count = finite_ic.sum(axis=0)
        ic_sum = np.where(finite_ic, ic_matrix, 0).sum(axis=0)
        ic_squared_sum = np.where(finite_ic, ic_matrix**2, 0).sum(axis=0)
        mean_ic = np.divide(ic_sum, valid_count, out=np.full(feature_count, np.nan), where=valid_count > 0)
        variance_ic = np.divide(
            ic_squared_sum, valid_count, out=np.full(feature_count, np.nan), where=valid_count > 0
        ) - mean_ic**2
        std_ic = np.sqrt(np.maximum(variance_ic, 0))
        positive_rate = np.divide(
            ((ic_matrix > 0) & finite_ic).sum(axis=0),
            valid_count,
            out=np.full(feature_count, np.nan),
            where=valid_count > 0,
        )
        for feature_index in range(feature_count):
            summary_rows.append(
                {
                    "split": split_name,
                    "feature": f"num_{feature_index}",
                    "feature_index": feature_index,
                    "mean_rank_ic": to_float(mean_ic[feature_index]),
                    "std_rank_ic": to_float(std_ic[feature_index]),
                    "positive_rate": to_float(positive_rate[feature_index]),
                }
            )
    return {"summary": summary_rows}


# ---------------------------------------------------------------- D5
def analyze_category(data, split_slices, mask_x, mask_y, T, category_count):
    cat_x = data["cat_x"]
    category_min = np.full(category_count, np.iinfo(np.int64).max, dtype=np.int64)
    category_max = np.full(category_count, np.iinfo(np.int64).min, dtype=np.int64)
    for chunk_start in range(0, T, 32):
        chunk_stop = min(chunk_start + 32, T)
        values = cat_x[chunk_start:chunk_stop][mask_x[chunk_start:chunk_stop]]
        if values.size:
            category_min = np.minimum(category_min, values.min(axis=0))
            category_max = np.maximum(category_max, values.max(axis=0))
    category_ranges = category_max - category_min + 1
    assert np.all(category_ranges < 2_000_000)

    category_counts = {}
    for split_name in SPLIT_NAMES:
        split_slice = split_slices[split_name]
        counts_by_feature = [
            np.zeros(int(category_ranges[feature_index]), dtype=np.int64)
            for feature_index in range(category_count)
        ]
        row_count = 0
        for chunk_start in range(split_slice.start, split_slice.stop, 32):
            chunk_stop = min(chunk_start + 32, split_slice.stop)
            selected_mask = mask_x[chunk_start:chunk_stop]
            if split_name != "pretrain":
                selected_mask = selected_mask & mask_y[chunk_start:chunk_stop]
            values = cat_x[chunk_start:chunk_stop][selected_mask]
            if values.size == 0:
                continue
            row_count += values.shape[0]
            for feature_index in range(category_count):
                counts_by_feature[feature_index] += np.bincount(
                    values[:, feature_index] - category_min[feature_index],
                    minlength=int(category_ranges[feature_index]),
                )
        category_counts[split_name] = {"row_count": row_count, "counts": counts_by_feature}

    profile_rows = []
    for feature_index in range(category_count):
        train_counts = category_counts["train"]["counts"][feature_index]
        for split_name in SPLIT_NAMES:
            split_counts = category_counts[split_name]["counts"][feature_index]
            row_count = category_counts[split_name]["row_count"]
            nonzero_counts = split_counts[split_counts > 0]
            unseen_count = split_counts[train_counts == 0].sum() if split_name != "train" else 0
            profile_rows.append(
                {
                    "category_feature": f"cat_{feature_index}",
                    "feature_index": feature_index,
                    "split": split_name,
                    "rows": int(row_count),
                    "min_code": int(category_min[feature_index]),
                    "max_code": int(category_max[feature_index]),
                    "cardinality": int(nonzero_counts.size),
                    "top1_share": float(nonzero_counts.max() / max(row_count, 1)),
                    "top5_share": float(np.sort(nonzero_counts)[-5:].sum() / max(row_count, 1)),
                    "rare_share_count_lt100": float(
                        nonzero_counts[nonzero_counts < 100].sum() / max(row_count, 1)
                    ),
                    "unseen_vs_train_rate": float(unseen_count / max(row_count, 1)),
                }
            )

    change_rows = []
    for split_name in SPLIT_NAMES:
        split_slice = split_slices[split_name]
        changed_counts = np.zeros(category_count, dtype=np.int64)
        eligible_pairs = 0
        for time_idx in range(max(split_slice.start + 1, 1), split_slice.stop):
            eligible_mask = mask_x[time_idx] & mask_x[time_idx - 1]
            eligible_count = np.count_nonzero(eligible_mask)
            eligible_pairs += eligible_count
            if eligible_count:
                changed_counts += np.count_nonzero(
                    cat_x[time_idx, eligible_mask] != cat_x[time_idx - 1, eligible_mask],
                    axis=0,
                )
        for feature_index in range(category_count):
            change_rows.append(
                {
                    "category_feature": f"cat_{feature_index}",
                    "split": split_name,
                    "change_rate": float(changed_counts[feature_index] / max(eligible_pairs, 1)),
                }
            )

    train_rows = int(category_counts["train"]["row_count"])
    total_onehot_columns = int(sum(
        row["cardinality"] for row in profile_rows if row["split"] == "train"
    ))
    estimated_nonzero_values = train_rows * category_count
    estimated_csr_mb = (estimated_nonzero_values * (4 + 4) + (train_rows + 1) * 4) / 1024**2

    encoding_plan = [
        {
            "feature_group": "cat_0,1,2,3,4,6,7,8",
            "profile": "低基数类别特征",
            "recommended_encoding": "稀疏 One-Hot",
            "important_setting": "handle_unknown='ignore'",
        },
        {
            "feature_group": "cat_5",
            "profile": "高基数、近似身份标识的特征",
            "recommended_encoding": "比较稀疏 One-Hot / 原生类别 / Embedding",
            "important_setting": "保留未知类别桶，禁止使用密集矩阵",
        },
        {
            "feature_group": "num_x",
            "profile": "已近似标准化，部分特征存在厚尾和漂移",
            "recommended_encoding": "原始值基线，比较排名或截尾实验",
            "important_setting": "所有阈值只在训练期拟合",
        },
    ]

    return {
        "profile": profile_rows,
        "change": change_rows,
        "onehot": {
            "train_rows": train_rows,
            "onehot_columns": total_onehot_columns,
            "non_zero_values": estimated_nonzero_values,
            "csr_mb": float(estimated_csr_mb),
        },
        "encoding_plan": encoding_plan,
    }


# ---------------------------------------------------------------- chart-ready 派生数据
def build_chart_payloads(numeric, rankic, category, label):
    features = numeric["features"]
    rankic_summary = rankic["summary"]

    def top_n(rows, key, n=TOP_N, absolute=False):
        ordered = sorted(
            rows,
            key=lambda row: (abs(row[key]) if absolute else row[key]),
            reverse=True,
        )[:n]
        return [{"feature": row["feature"], "value": float(row[key])} for row in ordered]

    top_max_abs = top_n(features, "max_abs")
    top_outlier_train = top_n(features, "train_outlier_rate")
    top_shift_valid = top_n(features, "valid_mean_shift_sd", absolute=True)
    top_psi_test = top_n(features, "test_psi")

    valid_ic = [row for row in rankic_summary if row["split"] == "valid"]
    train_ic = {row["feature"]: row for row in rankic_summary if row["split"] == "train_sample"}
    for row in valid_ic:
        row["abs_mean_rank_ic"] = abs(row["mean_rank_ic"])
    top_valid_ic = sorted(valid_ic, key=lambda row: row["abs_mean_rank_ic"], reverse=True)[:TOP_N]
    rankic_top_valid = [
        {
            "feature": row["feature"],
            "valid": float(row["mean_rank_ic"]),
            "train_sample": float(train_ic[row["feature"]]["mean_rank_ic"]),
        }
        for row in top_valid_ic
    ]

    # 全特征 |RankIC|（验证集）分布直方图；无截面方差的特征为 NaN，先过滤。
    def finite_histogram(values, bins=20):
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return {"edges": [], "counts": []}
        counts, edges = np.histogram(finite_values, bins=bins)
        return {"edges": edges.tolist(), "counts": counts.tolist()}

    rankic_dist = finite_histogram([abs(row["mean_rank_ic"]) for row in valid_ic])

    # 全特征漂移分布直方图
    max_psi_arr = np.asarray([row["max_psi"] for row in features])
    max_shift_arr = np.asarray([row["max_abs_mean_shift"] for row in features])
    drift_dist = {
        "max_psi": finite_histogram(max_psi_arr),
        "max_shift": finite_histogram(max_shift_arr),
    }

    # 类别派生数据
    cat_profile = category["profile"]
    train_cat = [row for row in cat_profile if row["split"] == "train"]
    cat_cardinality = [
        {"feature": row["category_feature"], "value": row["cardinality"]}
        for row in sorted(train_cat, key=lambda row: row["feature_index"])
    ]
    cat_unseen = {}
    for row in cat_profile:
        if row["split"] in ("valid", "test"):
            cat_unseen.setdefault(row["category_feature"], {})[row["split"]] = row["unseen_vs_train_rate"]
    cat_unseen_list = [
        {
            "feature": feature,
            "valid": cat_unseen.get(feature, {}).get("valid", 0.0),
            "test": cat_unseen.get(feature, {}).get("test", 0.0),
        }
        for feature in [row["category_feature"] for row in sorted(train_cat, key=lambda row: row["feature_index"])]
    ]
    cat_change = {}
    for row in category["change"]:
        cat_change.setdefault(row["category_feature"], {})[row["split"]] = row["change_rate"]
    cat_change_list = [
        {
            "feature": feature,
            "pretrain": cat_change.get(feature, {}).get("pretrain", 0.0),
            "train": cat_change.get(feature, {}).get("train", 0.0),
            "valid": cat_change.get(feature, {}).get("valid", 0.0),
            "test": cat_change.get(feature, {}).get("test", 0.0),
        }
        for feature in [row["category_feature"] for row in sorted(train_cat, key=lambda row: row["feature_index"])]
    ]

    return {
        "top_max_abs": top_max_abs,
        "top_outlier_train": top_outlier_train,
        "top_shift_valid": top_shift_valid,
        "top_psi_test": top_psi_test,
        "rankic_top_valid": rankic_top_valid,
        "rankic_dist": rankic_dist,
        "drift_dist": drift_dist,
        "cat_cardinality": cat_cardinality,
        "cat_unseen": cat_unseen_list,
        "cat_change": cat_change_list,
        "label_histogram": label["histogram"],
    }


# ---------------------------------------------------------------- selftest
def run_selftest(data, split_slices, mask_x, mask_y, coverage, feature_count):
    results = []
    ok_all = True

    def check(name, ok, detail=""):
        nonlocal ok_all
        ok_all = ok_all and ok
        results.append((name, "PASS" if ok else "FAIL", detail))

    # 覆盖率锚点
    by_split = {row["split"]: row for row in coverage["by_split"]}
    for split_name, anchors in PHASE1_COVERAGE_ANCHORS.items():
        for metric, anchor in anchors.items():
            actual = by_split[split_name][metric]
            ok = abs(actual - anchor) <= 1e-4
            check(f"anchor {split_name}.{metric}", ok, f"actual={actual:.6f} expected={anchor:.6f}")

    # 语义缺失率（全特征）与 target 原始缺失率
    total_positions = data["num_x"].shape[0] * data["num_x"].shape[1]
    semantic_missing = float((~mask_x).mean())
    target_missing = float((~np.isfinite(data["y1"])).mean())
    check(
        "semantic_missing_rate",
        abs(semantic_missing - PHASE1_SEMANTIC_MISSING_ANCHOR) <= 1e-4,
        f"actual={semantic_missing:.6f} expected={PHASE1_SEMANTIC_MISSING_ANCHOR:.6f}",
    )
    check(
        "target_missing_rate",
        abs(target_missing - PHASE1_TARGET_MISSING_ANCHOR) <= 1e-4,
        f"actual={target_missing:.6f} expected={PHASE1_TARGET_MISSING_ANCHOR:.6f}",
    )

    # 占位验证
    for row in coverage["placeholder"]:
        check(
            f"placeholder_zero {row['split']}",
            np.allclose(row["numeric_zero_rate"], 1.0) and np.allclose(row["allzero_row_rate"], 1.0),
            f"zero={row['numeric_zero_rate']:.6f} allzero={row['allzero_row_rate']:.6f}",
        )

    # Y1 概况
    y1 = data["y1"]
    for split_name in ("train", "valid"):
        split_slice = split_slices[split_name]
        usable = mask_x[split_slice] & mask_y[split_slice] & np.isfinite(y1[split_slice])
        values = y1[split_slice][usable].astype(np.float64)
        check(
            f"y1_{split_name}_mean",
            abs(values.mean() - 0.5) <= 0.01,
            f"mean={values.mean():.6f}",
        )
        check(
            f"y1_{split_name}_std",
            abs(values.std() - 0.289) <= 0.01,
            f"std={values.std():.6f}",
        )

    print("=" * 78)
    print("SELFTEST vs phase1 anchors")
    print("=" * 78)
    for name, status, detail in results:
        print(f"  [{status}] {name}: {detail}")
    print("=" * 78)
    print("ALL PASS" if ok_all else "SOME FAILED")
    print(f"total_positions={total_positions:,}")
    return ok_all


def main() -> int:
    parser = argparse.ArgumentParser(description="面向动态股票池的因果多模型排序预测系统：Y1 数据分析执行脚本")
    parser.add_argument("--seed", type=int, default=ANALYSIS_SEED, help="随机种子")
    parser.add_argument("--selftest", action="store_true", help="运行自检后退出")
    args = parser.parse_args()

    started_total = time.perf_counter()
    data, source, load_seconds = load_data()
    print(f"数据加载完成：{source}（{load_seconds:.2f} 秒）")

    train_start, valid_start, test_start, feature_count = validate_data(data)
    split_slices = make_split_slices(train_start, valid_start, test_start, data["num_x"].shape[0])
    mask_x = data["mask_x"]
    mask_y = data["mask_y"]
    T, stock_count, _ = data["num_x"].shape
    category_count = data["cat_x"].shape[2]
    rng = np.random.default_rng(args.seed)

    print(f"核心形状：T={T}, stocks={stock_count}, numeric={feature_count}, categorical={category_count}")
    print(f"官方切分：train=[{train_start},{valid_start}) valid=[{valid_start},{test_start}) test=[{test_start},{T})")
    print("开始分析……")

    # D1
    t0 = time.perf_counter()
    coverage = analyze_coverage(data, split_slices, T, stock_count)
    print(f"D1 掩码覆盖完成：{time.perf_counter() - t0:.1f}s")

    # D2
    t0 = time.perf_counter()
    label = analyze_label(data, split_slices, mask_x, mask_y, rng)
    print(f"D2 标签分析完成：{time.perf_counter() - t0:.1f}s")

    # D3
    t0 = time.perf_counter()
    numeric = analyze_numeric(data, split_slices, mask_x, mask_y, rng, feature_count)
    print(f"D3 数值特征完成：{time.perf_counter() - t0:.1f}s")

    # D4
    t0 = time.perf_counter()
    rankic = analyze_rank_ic(data, split_slices, mask_x, mask_y, feature_count)
    print(f"D4 RankIC 完成：{time.perf_counter() - t0:.1f}s")

    # D5
    t0 = time.perf_counter()
    category = analyze_category(data, split_slices, mask_x, mask_y, T, category_count)
    print(f"D5 类别特征完成：{time.perf_counter() - t0:.1f}s")

    if args.selftest:
        ok = run_selftest(data, split_slices, mask_x, mask_y, coverage, feature_count)
        return 0 if ok else 1

    chart_payloads = build_chart_payloads(numeric, rankic, category, label)

    results = {
        "meta": {
            "seed": args.seed,
            "source": source,
            "load_seconds": load_seconds,
            "dimensions": {"time": T, "stock": stock_count, "numeric": feature_count, "category": category_count},
            "splits": {
                "train": [train_start, valid_start],
                "valid": [valid_start, test_start],
                "test": [test_start, T],
            },
            "total_positions": int(T * stock_count),
        },
        "coverage": coverage,
        "label": label,
        "numeric": numeric,
        "rankic": rankic,
        "category": category,
        "charts": chart_payloads,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "analysis_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=1)

    # CSV 产物（供人工抽查）
    def write_csv(name, rows):
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")

    write_csv("mask_coverage.csv", coverage["by_split"])
    write_csv("label_profile.csv", label["profile"])
    write_csv("label_time_series.csv", label["time_rows"])
    write_csv("numeric_feature_profile.csv", numeric["features"])
    write_csv("rankic_summary.csv", rankic["summary"])
    write_csv("category_profile.csv", category["profile"])
    write_csv("category_change.csv", category["change"])

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seed": args.seed,
        "data_source": source,
        "load_seconds": load_seconds,
        "total_runtime_seconds": time.perf_counter() - started_total,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "outputs": [
            "analysis_results.json",
            "mask_coverage.csv",
            "label_profile.csv",
            "label_time_series.csv",
            "numeric_feature_profile.csv",
            "rankic_summary.csv",
            "category_profile.csv",
            "category_change.csv",
        ],
    }
    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"产物已写入：{OUTPUT_DIR}")
    print(f"总耗时：{time.perf_counter() - started_total:.1f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
