from __future__ import annotations

"""Ordered stock-identity residual calibration with strict temporal cutoffs.

The diagnostic reuses existing strict-OOF and frozen official-Valid predictions.
It does not retrain any model.  Per the explicit user artifact override recorded
in protocol.json, an isolated Test prediction is generated even when promotion
gates fail; a failed candidate is marked evidence-only and is never submitted or
copied to final_submission.
"""

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "02_experiments"))

from exp_016_unified_expert_fusion.config import BASE_WEIGHTS, FAMILIES  # noqa: E402
from exp_016_unified_expert_fusion.src.prediction_contract import validate_prediction  # noqa: E402


COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
OOF_PATH = ROOT / "03_cache" / "exp_016_unified_expert_fusion" / "oof_predictions" / "family_matrix.npy"
VALID_BASE_PATH = ROOT / "04_results" / "_acceptance" / "exp021_validation_audit" / "full_valid_prediction.npy"
VALID_DRIFT_PATH = ROOT / "04_results" / "_acceptance" / "exp021_validation_audit" / "per_time_ic.csv"
TEST_BASE_PATH = ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy"
RESULT = ROOT / "04_results" / "exp_028a_ordered_entity_residual"
PROTOCOL = RESULT / "protocol.json"

TRAIN_START = 486
OOF_START = 1459
FOLD_1_STOP = 1945
FOLD_2_STOP = 2432
OOF_STOP = 2918
VALID_START = 2918
VALID_STOP = 3161
TEST_START = 3161
TEST_STOP = 3603
STOCK_COUNT = 5282
OOF_CAP = 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    ranked = rankdata(array, method="average")
    return np.asarray((ranked - 1.0) / max(1, ranked.size - 1), dtype=np.float32)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
    return value if np.isfinite(value) else 0.0


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def capped_positions(groups: np.ndarray, start_time: int, stop_time: int) -> Iterable[tuple[int, np.ndarray]]:
    offsets = np.concatenate([[0], np.cumsum(np.asarray(groups, dtype=np.int64))])
    for time_index in range(start_time, stop_time):
        group_index = time_index - TRAIN_START
        left, right = int(offsets[group_index]), int(offsets[group_index + 1])
        size = right - left
        take = min(size, OOF_CAP)
        if take != OOF_CAP:
            raise RuntimeError(f"Expected {OOF_CAP} rows at time {time_index}, got {take}")
        yield time_index, left + np.linspace(0, size - 1, take, dtype=np.int64)


def load_oof_panel() -> dict[str, np.ndarray]:
    groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    train_stock = np.load(COMMON / "train_stock.npy", mmap_mode="r")
    family_matrix = np.load(OOF_PATH, mmap_mode="r")
    section_count = OOF_STOP - OOF_START
    expected_shape = (section_count * OOF_CAP, len(FAMILIES))
    if family_matrix.shape != expected_shape:
        raise RuntimeError(f"Unexpected OOF matrix shape {family_matrix.shape}; expected {expected_shape}")

    weights = np.asarray([BASE_WEIGHTS[name] for name in FAMILIES], dtype=np.float64)
    weights /= weights.sum()
    base = np.empty((section_count, OOF_CAP), dtype=np.float32)
    target = np.empty_like(base)
    stocks = np.empty((section_count, OOF_CAP), dtype=np.int32)
    times = np.arange(OOF_START, OOF_STOP, dtype=np.int32)

    for section_index, (time_index, absolute) in enumerate(capped_positions(groups, OOF_START, OOF_STOP)):
        block = np.asarray(
            family_matrix[section_index * OOF_CAP:(section_index + 1) * OOF_CAP],
            dtype=np.float32,
        )
        ranked_families = np.stack([rank01(block[:, column]) for column in range(block.shape[1])], axis=1)
        base[section_index] = rank01(ranked_families @ weights)
        target[section_index] = rank01(np.asarray(train_y[absolute], dtype=np.float32))
        stocks[section_index] = np.asarray(train_stock[absolute], dtype=np.int32)
        if int(times[section_index]) != time_index:
            raise RuntimeError("OOF time alignment failure")
    return {"times": times, "stocks": stocks, "base": base, "target": target}


def load_valid_panel() -> dict[str, Any]:
    groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    stocks = np.asarray(np.load(COMMON / "valid_stock.npy", mmap_mode="r"), dtype=np.int32)
    target = np.asarray(np.load(COMMON / "valid_y.npy", mmap_mode="r"), dtype=np.float32)
    base = np.asarray(np.load(VALID_BASE_PATH, mmap_mode="r"), dtype=np.float32)
    if not (stocks.size == target.size == base.size == int(groups.sum())):
        raise RuntimeError("Official Valid panel alignment failure")
    return {
        "times": np.arange(VALID_START, VALID_STOP, dtype=np.int32),
        "groups": groups,
        "stocks": stocks,
        "base": base,
        "target": target,
    }


def estimate_offsets(stocks: np.ndarray, residuals: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    stocks = np.asarray(stocks, dtype=np.int32).reshape(-1)
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if stocks.size != residuals.size or not np.isfinite(residuals).all():
        raise RuntimeError("Invalid calibration residual panel")

    counts = np.bincount(stocks, minlength=STOCK_COUNT).astype(np.int64)
    sums = np.bincount(stocks, weights=residuals, minlength=STOCK_COUNT)
    sums2 = np.bincount(stocks, weights=residuals * residuals, minlength=STOCK_COUNT)
    observed = counts > 0
    means = np.zeros(STOCK_COUNT, dtype=np.float64)
    means[observed] = sums[observed] / counts[observed]

    sample_variance = np.zeros(STOCK_COUNT, dtype=np.float64)
    repeated = counts > 1
    numerator = sums2[repeated] - (sums[repeated] * sums[repeated]) / counts[repeated]
    sample_variance[repeated] = np.maximum(numerator / (counts[repeated] - 1), 0.0)
    global_variance = float(np.var(residuals, ddof=1)) if residuals.size > 1 else 0.0
    sample_variance[observed & ~repeated] = global_variance
    mean_noise = np.zeros(STOCK_COUNT, dtype=np.float64)
    mean_noise[observed] = sample_variance[observed] / counts[observed]

    observed_means = means[observed]
    observed_noise = mean_noise[observed]
    between_raw = float(np.var(observed_means, ddof=1)) if observed_means.size > 1 else 0.0
    tau2 = max(between_raw - float(np.mean(observed_noise)), 0.0)
    shrinkage = np.zeros(STOCK_COUNT, dtype=np.float64)
    if tau2 > 0.0:
        shrinkage[observed] = tau2 / (tau2 + mean_noise[observed])
    offsets = np.asarray(shrinkage * means, dtype=np.float32)
    offsets[~observed] = 0.0
    details = {
        "counts": counts,
        "raw_mean": means,
        "mean_noise_variance": mean_noise,
        "shrinkage": shrinkage,
        "offset": offsets,
    }
    summary = {
        "calibration_rows": int(stocks.size),
        "observed_entities": int(observed.sum()),
        "unseen_entities": int((~observed).sum()),
        "global_residual_variance": global_variance,
        "raw_between_entity_mean_variance": between_raw,
        "estimated_between_entity_variance_tau2": tau2,
        "mean_abs_offset": float(np.mean(np.abs(offsets[observed]))) if observed.any() else 0.0,
        "max_abs_offset": float(np.max(np.abs(offsets))) if observed.any() else 0.0,
        "mean_shrinkage": float(np.mean(shrinkage[observed])) if observed.any() else 0.0,
    }
    return offsets, details, summary


def evaluate_fixed_sections(
    window: str,
    times: np.ndarray,
    stocks: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    offsets: np.ndarray,
    calibration_cutoff_exclusive: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    deltas = np.empty(times.size, dtype=np.float64)
    correlations = np.empty(times.size, dtype=np.float64)
    for index, time_index in enumerate(times):
        base_rank = rank01(base[index])
        candidate = rank01(base_rank + offsets[stocks[index]])
        target_rank = rank01(target[index])
        base_ic = safe_corr(base_rank, target_rank)
        candidate_ic = safe_corr(candidate, target_rank)
        deltas[index] = candidate_ic - base_ic
        correlations[index] = safe_corr(base_rank, candidate)
        rows.append({
            "window": window,
            "time": int(time_index),
            "baseline_ic": base_ic,
            "candidate_ic": candidate_ic,
            "delta_vs_baseline": float(deltas[index]),
            "rank_correlation_to_baseline": float(correlations[index]),
            "calibration_cutoff_exclusive": calibration_cutoff_exclusive,
            "in_window_label_update": False,
        })
    return rows, deltas, correlations


def evaluate_variable_sections(
    window: str,
    times: np.ndarray,
    groups: np.ndarray,
    stocks: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    offsets: np.ndarray,
    calibration_cutoff_exclusive: int,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    deltas = np.empty(groups.size, dtype=np.float64)
    correlations = np.empty(groups.size, dtype=np.float64)
    residual = np.empty_like(base, dtype=np.float32)
    position = 0
    for index, size_value in enumerate(groups):
        size = int(size_value)
        left, right = position, position + size
        current_stocks = stocks[left:right]
        base_rank = rank01(base[left:right])
        target_rank = rank01(target[left:right])
        candidate = rank01(base_rank + offsets[current_stocks])
        residual[left:right] = target_rank - base_rank
        base_ic = safe_corr(base_rank, target_rank)
        candidate_ic = safe_corr(candidate, target_rank)
        deltas[index] = candidate_ic - base_ic
        correlations[index] = safe_corr(base_rank, candidate)
        rows.append({
            "window": window,
            "time": int(times[index]),
            "baseline_ic": base_ic,
            "candidate_ic": candidate_ic,
            "delta_vs_baseline": float(deltas[index]),
            "rank_correlation_to_baseline": float(correlations[index]),
            "calibration_cutoff_exclusive": calibration_cutoff_exclusive,
            "in_window_label_update": False,
        })
        position = right
    return rows, deltas, correlations, residual


def shuffled_null_fixed(stocks: np.ndarray, base: np.ndarray, target: np.ndarray, offsets: np.ndarray,
                        repetitions: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        shuffled = offsets[rng.permutation(STOCK_COUNT)]
        deltas = []
        for index in range(base.shape[0]):
            base_rank = rank01(base[index])
            target_rank = rank01(target[index])
            candidate = rank01(base_rank + shuffled[stocks[index]])
            deltas.append(safe_corr(candidate, target_rank) - safe_corr(base_rank, target_rank))
        values[repetition] = float(np.mean(deltas))
    return values


def shuffled_null_variable(groups: np.ndarray, stocks: np.ndarray, base: np.ndarray, target: np.ndarray,
                           offsets: np.ndarray, repetitions: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(repetitions, dtype=np.float64)
    boundaries = np.concatenate([[0], np.cumsum(groups)])
    for repetition in range(repetitions):
        shuffled = offsets[rng.permutation(STOCK_COUNT)]
        deltas = []
        for index in range(groups.size):
            left, right = int(boundaries[index]), int(boundaries[index + 1])
            base_rank = rank01(base[left:right])
            target_rank = rank01(target[left:right])
            candidate = rank01(base_rank + shuffled[stocks[left:right]])
            deltas.append(safe_corr(candidate, target_rank) - safe_corr(base_rank, target_rank))
        values[repetition] = float(np.mean(deltas))
    return values


def contiguous_blocks(values: np.ndarray, block_size: int) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    return [array[left:left + block_size] for left in range(0, array.size, block_size) if array[left:left + block_size].size]


def block_bootstrap(window_values: list[np.ndarray], block_size: int, repetitions: int,
                    seed: int) -> dict[str, float]:
    blocks = [block for values in window_values for block in contiguous_blocks(values, block_size)]
    if not blocks:
        return {"mean": 0.0, "lower_90": 0.0, "upper_90": 0.0, "worst_block": 0.0}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        draws[repetition] = float(np.mean(np.concatenate([blocks[item] for item in selected])))
    pooled = np.concatenate(window_values)
    return {
        "mean": float(np.mean(pooled)),
        "lower_90": float(np.quantile(draws, 0.05)),
        "upper_90": float(np.quantile(draws, 0.95)),
        "worst_block": float(min(np.mean(block) for block in blocks)),
    }


def entity_persistence(offsets: np.ndarray, stocks: np.ndarray, residuals: np.ndarray) -> dict[str, float]:
    counts = np.bincount(stocks.reshape(-1), minlength=STOCK_COUNT)
    sums = np.bincount(stocks.reshape(-1), weights=residuals.reshape(-1), minlength=STOCK_COUNT)
    realized = np.zeros(STOCK_COUNT, dtype=np.float64)
    observed = counts > 0
    realized[observed] = sums[observed] / counts[observed]
    eligible = observed & np.isfinite(offsets) & (np.abs(offsets) > 0)
    correlation = spearmanr(offsets[eligible], realized[eligible]).statistic if eligible.sum() >= 3 else 0.0
    return {
        "eligible_entities": int(eligible.sum()),
        "spearman_prior_offset_vs_realized_residual": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def load_valid_drift() -> dict[int, float]:
    result: dict[int, float] = {}
    with VALID_DRIFT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result[int(row["time"])] = float(row["drift_score"])
    return result


def generate_test_prediction(offsets: np.ndarray, preserve: int) -> tuple[np.ndarray, dict[str, Any]]:
    test_groups = np.asarray(np.load(COMMON / "test_group_sizes.npy"), dtype=np.int64)
    test_times = np.asarray(np.load(COMMON / "test_time.npy", mmap_mode="r"), dtype=np.int32)
    test_stocks = np.asarray(np.load(COMMON / "test_stock.npy", mmap_mode="r"), dtype=np.int32)
    base_grid = np.asarray(np.load(TEST_BASE_PATH, mmap_mode="r"), dtype=np.float32)
    output = base_grid.copy()
    boundaries = np.concatenate([[0], np.cumsum(test_groups)])
    per_time_correlations = []
    changed_sections = 0
    for section_index in range(test_groups.size):
        left, right = int(boundaries[section_index]), int(boundaries[section_index + 1])
        time_index = int(test_times[left])
        stocks = test_stocks[left:right]
        base = base_grid[time_index - TEST_START, stocks]
        candidate = base.copy() if section_index < preserve else rank01(rank01(base) + offsets[stocks])
        output[time_index - TEST_START, stocks] = candidate.astype(np.float32)
        correlation = safe_corr(base, candidate)
        per_time_correlations.append(correlation)
        if not np.array_equal(base.astype(np.float32), candidate.astype(np.float32)):
            changed_sections += 1
    mask = np.zeros_like(output, dtype=bool)
    mask[test_times - TEST_START, test_stocks] = True
    contract = validate_prediction(output, mask)
    details = {
        "preserved_first_sections": preserve,
        "changed_sections": changed_sections,
        "evaluation_pearson_to_exp024b": float(np.corrcoef(base_grid[mask], output[mask])[0, 1]),
        "per_time_rank_correlation_mean": float(np.mean(per_time_correlations)),
        "per_time_rank_correlation_min": float(np.min(per_time_correlations)),
        "contract": contract,
    }
    return output, details


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    params = protocol["preregistered_parameters"]
    gates = protocol["decision_gates"]
    rng = np.random.default_rng(int(params["random_seed"]))

    protected = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
        "exp024b_prediction": TEST_BASE_PATH,
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    hashes_before = {name: sha256(path) for name, path in protected.items()}

    print("loading existing strict-OOF panel", flush=True)
    oof = load_oof_panel()
    valid = load_valid_panel()
    residual_oof = oof["target"] - oof["base"]
    fold1_end = FOLD_1_STOP - OOF_START
    fold2_end = FOLD_2_STOP - OOF_START

    window_specs = [
        ("fold_2", 0, fold1_end, fold1_end, fold2_end, FOLD_1_STOP),
        ("fold_3", 0, fold2_end, fold2_end, OOF_STOP - OOF_START, FOLD_2_STOP),
    ]
    per_time_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    window_results: dict[str, dict[str, Any]] = {}
    window_deltas: list[np.ndarray] = []
    all_correlations: list[np.ndarray] = []

    for window_name, fit_start, fit_stop, eval_start, eval_stop, cutoff in window_specs:
        print(f"[{window_name}] fit earlier residuals and evaluate frozen holdout", flush=True)
        offsets, _, estimator = estimate_offsets(
            oof["stocks"][fit_start:fit_stop], residual_oof[fit_start:fit_stop]
        )
        rows, deltas, correlations = evaluate_fixed_sections(
            window_name,
            oof["times"][eval_start:eval_stop],
            oof["stocks"][eval_start:eval_stop],
            oof["base"][eval_start:eval_stop],
            oof["target"][eval_start:eval_stop],
            offsets,
            cutoff,
        )
        null = shuffled_null_fixed(
            oof["stocks"][eval_start:eval_stop],
            oof["base"][eval_start:eval_stop],
            oof["target"][eval_start:eval_stop],
            offsets,
            int(params["shuffle_repetitions"]),
            rng,
        )
        persistence = entity_persistence(
            offsets,
            oof["stocks"][eval_start:eval_stop],
            residual_oof[eval_start:eval_stop],
        )
        per_time_rows.extend(rows)
        window_deltas.append(deltas)
        all_correlations.append(correlations)
        for repetition, value in enumerate(null):
            null_rows.append({"window": window_name, "repetition": repetition, "mean_delta_vs_baseline": float(value)})
        window_results[window_name] = {
            "sections": int(deltas.size),
            "mean_delta": float(np.mean(deltas)),
            "positive_delta_ratio": float(np.mean(deltas > 0)),
            "mean_rank_correlation_to_baseline": float(np.mean(correlations)),
            "shuffled_stock_p95": float(np.quantile(null, 0.95)),
            "above_shuffled_stock_p95": float(np.mean(deltas)) > float(np.quantile(null, 0.95)),
            "estimator": estimator,
            "entity_persistence": persistence,
        }

    print("[official_valid] fit all strict OOF residuals and evaluate frozen exp021", flush=True)
    offsets_valid, _, estimator_valid = estimate_offsets(oof["stocks"], residual_oof)
    valid_rows, valid_deltas, valid_correlations, residual_valid = evaluate_variable_sections(
        "official_valid",
        valid["times"],
        valid["groups"],
        valid["stocks"],
        valid["base"],
        valid["target"],
        offsets_valid,
        OOF_STOP,
    )
    valid_null = shuffled_null_variable(
        valid["groups"], valid["stocks"], valid["base"], valid["target"], offsets_valid,
        int(params["shuffle_repetitions"]), rng,
    )
    valid_persistence = entity_persistence(offsets_valid, valid["stocks"], residual_valid)
    per_time_rows.extend(valid_rows)
    window_deltas.append(valid_deltas)
    all_correlations.append(valid_correlations)
    for repetition, value in enumerate(valid_null):
        null_rows.append({"window": "official_valid", "repetition": repetition, "mean_delta_vs_baseline": float(value)})
    window_results["official_valid"] = {
        "sections": int(valid_deltas.size),
        "mean_delta": float(np.mean(valid_deltas)),
        "positive_delta_ratio": float(np.mean(valid_deltas > 0)),
        "mean_rank_correlation_to_baseline": float(np.mean(valid_correlations)),
        "shuffled_stock_p95": float(np.quantile(valid_null, 0.95)),
        "above_shuffled_stock_p95": float(np.mean(valid_deltas)) > float(np.quantile(valid_null, 0.95)),
        "estimator": estimator_valid,
        "entity_persistence": valid_persistence,
    }

    bootstrap = block_bootstrap(
        window_deltas,
        int(params["bootstrap_block_size"]),
        int(params["bootstrap_repetitions"]),
        int(params["random_seed"]),
    )
    pooled = np.concatenate(window_deltas)
    second_half = float(np.mean(pooled[pooled.size // 2:]))
    drift_by_time = load_valid_drift()
    drift = np.asarray([drift_by_time[int(time_index)] for time_index in valid["times"]], dtype=np.float64)
    threshold = float(np.quantile(drift, float(params["high_drift_quantile"])))
    high_drift_delta = float(np.mean(valid_deltas[drift >= threshold]))
    mean_correlation = float(np.mean(np.concatenate(all_correlations)))

    pass_checks = {
        "all_three_window_deltas_positive": all(result["mean_delta"] > 0 for result in window_results.values()),
        "official_valid_delta_at_least_0_0025": window_results["official_valid"]["mean_delta"] >= float(gates["official_valid_minimum_delta"]),
        "pooled_delta_at_least_0_0015": bootstrap["mean"] >= float(gates["pooled_minimum_delta"]),
        "bootstrap_90_lower_positive": bootstrap["lower_90"] > 0,
        "worst_32_section_block_at_least_minus_0_001": bootstrap["worst_block"] >= float(gates["minimum_worst_32_section_block"]),
        "second_half_nonnegative": second_half >= 0,
        "high_drift_nonnegative": high_drift_delta >= 0,
        "above_shuffled_stock_p95_every_window": all(result["above_shuffled_stock_p95"] for result in window_results.values()),
        "mean_rank_correlation_to_baseline_at_least_0_995": mean_correlation >= float(gates["minimum_mean_rank_correlation_to_baseline"]),
    }

    # Final Test calibration is permitted by the explicit artifact override.  It
    # adds official-Valid residuals only after their holdout evaluation is done.
    final_stocks = np.concatenate([oof["stocks"].reshape(-1), valid["stocks"]])
    final_residuals = np.concatenate([residual_oof.reshape(-1), residual_valid])
    final_offsets, final_details, final_estimator = estimate_offsets(final_stocks, final_residuals)
    print("generating required isolated prediction_1.npy", flush=True)
    prediction, prediction_details = generate_test_prediction(
        final_offsets, int(params["preserve_exp024b_first_sections"])
    )
    atomic_npy(RESULT / "prediction_1.npy", prediction)
    atomic_npy(RESULT / "entity_offsets.npy", final_offsets)
    prediction_hash = sha256(RESULT / "prediction_1.npy")
    atomic_text(RESULT / "prediction.sha256", prediction_hash + "  prediction_1.npy\n")

    hashes_after = {name: sha256(path) for name, path in protected.items()}
    pass_checks["protected_hashes_unchanged"] = hashes_before == hashes_after
    passed = bool(all(pass_checks.values()))
    decision = "passed_candidate_generated_not_submitted" if passed else "failed_gates_prediction_generated_evidence_only_not_promotable"

    entity_rows = []
    for stock_index in range(STOCK_COUNT):
        entity_rows.append({
            "stock": stock_index,
            "count": int(final_details["counts"][stock_index]),
            "raw_mean_residual": float(final_details["raw_mean"][stock_index]),
            "mean_noise_variance": float(final_details["mean_noise_variance"][stock_index]),
            "shrinkage": float(final_details["shrinkage"][stock_index]),
            "offset": float(final_details["offset"][stock_index]),
        })
    write_csv(RESULT / "per_time_metrics.csv", per_time_rows)
    write_csv(RESULT / "random_null_summary.csv", null_rows)
    write_csv(RESULT / "per_stock_residual_diagnostics.csv", entity_rows)

    metrics = {
        "experiment": "exp_028a_ordered_entity_residual",
        "decision": decision,
        "passed": passed,
        "changed_variable": protocol["changed_variable"],
        "parameters": params,
        "window_results": window_results,
        "pooled_bootstrap": bootstrap,
        "second_half_mean_delta": second_half,
        "official_valid_high_drift_threshold": threshold,
        "official_valid_high_drift_mean_delta": high_drift_delta,
        "mean_rank_correlation_to_baseline": mean_correlation,
        "pass_checks": pass_checks,
        "final_estimator": final_estimator,
        "prediction": "prediction_1.npy",
        "prediction_sha256": prediction_hash,
        "prediction_details": prediction_details,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_unchanged": hashes_before == hashes_after,
        "test_labels_loaded": False,
        "future_features_loaded": False,
        "model_training_performed": False,
        "prediction_generated_under_user_override": True,
        "automatic_online_submission": False,
        "formal_submission_overwritten": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    metadata = {
        "status": "completed_passed_candidate_not_submitted" if passed else "completed_failed_evidence_only_prediction_generated",
        "causal_status": "compliant_strict_ordered_residual_calibration",
        "decision": decision,
        "prediction": "prediction_1.npy",
        "prediction_sha256": prediction_hash,
        "formal_submission_overwritten": False,
        "test_labels_loaded": False,
        "online_submission_used": False,
        "user_artifact_override_applied": True,
    }
    readme = f"""# exp028a：时间有序个股残差收缩校准

状态：`{metadata['status']}`。

唯一变化是：在冻结基线截面秩上加入仅由更早严格 OOF 残差估计的个股经验贝叶斯收缩偏置。没有训练新模型、没有窗口内标签更新、没有未来特征或 Test 标签。

## 诊断结果

- fold2 delta：`{window_results['fold_2']['mean_delta']:+.6f}`；
- fold3 delta：`{window_results['fold_3']['mean_delta']:+.6f}`；
- official Valid delta：`{window_results['official_valid']['mean_delta']:+.6f}`；
- pooled delta：`{bootstrap['mean']:+.6f}`，90% bootstrap 下界 `{bootstrap['lower_90']:+.6f}`；
- 最差32截面块：`{bootstrap['worst_block']:+.6f}`；
- 决策：`{decision}`。

## 强制独立预测产物

用户明确要求无论门槛结果均生成 `prediction_1.npy`。本文件 SHA-256 为 `{prediction_hash}`，shape `{tuple(prediction.shape)}`，dtype `{prediction.dtype}`。前 `{params['preserve_exp024b_first_sections']}` 个锚点截面逐值保留 exp024b，其余截面应用冻结个股偏置。

门槛失败时该预测仅为 `evidence_only_not_promotable`；不会自动提交，也不会覆盖 `final_submission`。
"""
    atomic_json(RESULT / "metrics.json", metrics)
    atomic_json(RESULT / "metadata.json", metadata)
    atomic_text(RESULT / "README.md", readme)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0 if prediction_details["contract"]["finite"] and hashes_before == hashes_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
