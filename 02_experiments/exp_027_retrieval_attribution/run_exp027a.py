from __future__ import annotations

"""Strictly causal attribution of the exp024b retrieval correction.

This diagnostic never opens Test arrays and never writes a prediction.  It
decomposes the normalized retrieval fingerprint into a global historical prior
and a state-specific residual, then evaluates fixed controls on the exact
exp024a walk-forward windows.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP024 = ROOT / "02_experiments" / "exp_024_state_retrieved_rank_residual"
sys.path.insert(0, str(EXP024))

from run_exp024a import (  # noqa: E402
    COMMON,
    EVAL_WINDOWS,
    OOF_START,
    RESULT as EXP024A_RESULT,
    TREE,
    VALID_BASE,
    build_panel,
    capped_positions,
    correction_score,
    feature_contract,
    oof_proxy_parts,
    rank01,
    retrieval_coordinates,
    safe_corr,
    sha256,
)


RESULT = ROOT / "04_results" / "exp_027a_retrieval_attribution"
PROTOCOL = RESULT / "protocol.json"


def normalized_fingerprint(fingerprint: np.ndarray) -> np.ndarray:
    value = np.asarray(fingerprint, dtype=np.float64)
    scale = float(np.sum(np.abs(value)))
    return value / scale if scale > 1e-12 else np.zeros_like(value)


def candidate_from_score(base_rank: np.ndarray, score: np.ndarray, alpha: float) -> np.ndarray:
    return rank01((1.0 - alpha) * base_rank + alpha * rank01(score))


def contiguous_blocks(values: np.ndarray, block_size: int) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    return [array[left:left + block_size] for left in range(0, array.size, block_size) if array[left:left + block_size].size]


def block_bootstrap(values: np.ndarray, block_size: int, repetitions: int, seed: int) -> dict[str, float]:
    blocks = contiguous_blocks(values, block_size)
    if not blocks:
        return {"mean": 0.0, "lower_90": 0.0, "upper_90": 0.0, "worst_block": 0.0}
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        draws[index] = float(np.mean(np.concatenate([blocks[item] for item in selected])))
    return {
        "mean": float(np.mean(values)),
        "lower_90": float(np.quantile(draws, 0.05)),
        "upper_90": float(np.quantile(draws, 0.95)),
        "worst_block": float(min(np.mean(block) for block in blocks)),
    }


def read_exp024a_retrieval() -> dict[tuple[str, int], dict[str, float]]:
    reference: dict[tuple[str, int], dict[str, float]] = {}
    with (EXP024A_RESULT / "per_time_metrics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "retrieval":
                reference[(row["window"], int(row["time"]))] = {
                    "candidate_ic": float(row["candidate_ic"]),
                    "delta": float(row["delta"]),
                }
    return reference


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    params = protocol["preregistered_parameters"]
    gates = protocol["decision_gates"]
    k = int(params["neighbors"])
    alpha = float(params["alpha"])
    components = int(params["state_pca_components"])
    random_repetitions = int(params["random_repetitions"])
    random_seed = int(params["random_seed"])

    protected = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
        "exp024b_prediction": ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy",
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    hashes_before = {name: sha256(path) for name, path in protected.items()}

    feature_positions, feature_names = feature_contract()
    if len(feature_positions) != int(params["expected_feature_count"]):
        raise RuntimeError(f"Feature contract changed: {len(feature_positions)}")
    train = build_panel("train", feature_positions)
    valid = build_panel("valid", feature_positions)
    proxy = oof_proxy_parts(train["groups"])
    valid_base = np.load(VALID_BASE, mmap_mode="r")
    train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
    valid_x = np.load(TREE / "valid_X.npy", mmap_mode="r")
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    train_positions = capped_positions(train["groups"])
    valid_offsets = np.concatenate([[0], np.cumsum(valid["groups"])])

    per_time_rows: list[dict[str, Any]] = []
    random_values: dict[tuple[str, int], list[float]] = {}
    decomposition_error = 0.0

    for window_index, (window_name, split, start, stop, baseline_name) in enumerate(EVAL_WINDOWS):
        library_mask = train["times"] < start
        library_states = train["states"][library_mask]
        library_fp = train["fingerprints"][library_mask]
        library_times = train["times"][library_mask]
        panel = train if split == "train" else valid
        query_mask = (panel["times"] >= start) & (panel["times"] < stop)
        query_indices = np.flatnonzero(query_mask)
        library_coord, query_coord = retrieval_coordinates(library_states, panel["states"][query_mask], components)
        recent = np.arange(max(0, library_fp.shape[0] - k), library_fp.shape[0])
        global_fp = np.mean(library_fp, axis=0)
        print(f"[{window_name}] library={library_fp.shape[0]} query={query_indices.size}", flush=True)

        for local_index, group_index in enumerate(query_indices):
            time_index = int(panel["times"][group_index])
            distances = np.sqrt(np.sum((library_coord - query_coord[local_index]) ** 2, axis=1))
            nearest = np.argpartition(distances, k - 1)[:k]
            retrieval_fp = np.mean(library_fp[nearest], axis=0)
            recent_fp = np.mean(library_fp[recent], axis=0)

            if split == "train":
                absolute = train_positions[group_index]
                x = np.asarray(train_x[absolute[:, None], np.asarray(feature_positions)[None, :]], dtype=np.float32)
                y = np.asarray(train_y[absolute], dtype=np.float32)
                base = proxy[time_index]
            else:
                left, right = int(valid_offsets[group_index]), int(valid_offsets[group_index + 1])
                x = np.asarray(valid_x[left:right, feature_positions], dtype=np.float32)
                y = np.asarray(valid_y[left:right], dtype=np.float32)
                base = np.asarray(valid_base[left:right], dtype=np.float32)

            base_rank = rank01(base)
            y_rank = rank01(y)
            base_ic = safe_corr(base_rank, y_rank)
            retrieval_score = correction_score(x, retrieval_fp)
            global_score = correction_score(x, global_fp)
            recent_score = correction_score(x, recent_fp)
            residual_score = np.asarray(retrieval_score, dtype=np.float64) - np.asarray(global_score, dtype=np.float64)

            w_retrieval = normalized_fingerprint(retrieval_fp)
            w_global = normalized_fingerprint(global_fp)
            w_residual = w_retrieval - w_global
            error = float(np.max(np.abs(w_retrieval - (w_global + w_residual))))
            decomposition_error = max(decomposition_error, error)

            scores = {
                "retrieval": retrieval_score,
                "global": global_score,
                "recent": recent_score,
                "residual": residual_score,
            }
            for method, score in scores.items():
                candidate = candidate_from_score(base_rank, score, alpha)
                candidate_ic = safe_corr(candidate, y_rank)
                per_time_rows.append({
                    "window": window_name,
                    "split": split,
                    "time": time_index,
                    "baseline": baseline_name,
                    "method": method,
                    "baseline_ic": base_ic,
                    "correction_ic": safe_corr(rank01(score), y_rank),
                    "candidate_ic": candidate_ic,
                    "delta_vs_baseline": candidate_ic - base_ic,
                    "mean_neighbor_distance": float(np.mean(distances[nearest])),
                    "min_library_time": int(np.min(library_times[nearest])) if method == "retrieval" else int(np.min(library_times)),
                    "max_library_time": int(np.max(library_times[nearest])) if method == "retrieval" else int(np.max(library_times)),
                    "library_cutoff_exclusive": start,
                })

            for repetition in range(random_repetitions):
                seed = random_seed + window_index * 10_000_019 + repetition * 1_000_003 + time_index
                picked = np.random.default_rng(seed).choice(library_fp.shape[0], size=k, replace=False)
                random_score = correction_score(x, np.mean(library_fp[picked], axis=0))
                random_candidate = candidate_from_score(base_rank, random_score, alpha)
                random_ic = safe_corr(random_candidate, y_rank)
                random_values.setdefault((window_name, repetition), []).append(random_ic - base_ic)

    by_key = {(row["window"], int(row["time"]), row["method"]): row for row in per_time_rows}
    summaries: list[dict[str, Any]] = []
    for window_name, _, start, stop, baseline_name in EVAL_WINDOWS:
        for method in ("retrieval", "global", "recent", "residual"):
            selected = [row for row in per_time_rows if row["window"] == window_name and row["method"] == method]
            delta = np.asarray([row["delta_vs_baseline"] for row in selected], dtype=np.float64)
            summaries.append({
                "window": window_name,
                "baseline": baseline_name,
                "method": method,
                "sections": len(selected),
                "mean_delta_vs_baseline": float(np.mean(delta)),
                "positive_delta_ratio": float(np.mean(delta > 0)),
                "mean_candidate_ic": float(np.mean([row["candidate_ic"] for row in selected])),
            })

    random_rows: list[dict[str, Any]] = []
    random_p95: dict[str, float] = {}
    for window_name, _, _, _, _ in EVAL_WINDOWS:
        means = []
        for repetition in range(random_repetitions):
            values = np.asarray(random_values[(window_name, repetition)], dtype=np.float64)
            mean_delta = float(np.mean(values))
            means.append(mean_delta)
            random_rows.append({"window": window_name, "repetition": repetition, "mean_delta_vs_baseline": mean_delta})
        random_p95[window_name] = float(np.quantile(means, 0.95))

    reference = read_exp024a_retrieval()
    reproduction_errors = []
    for key, expected in reference.items():
        actual = by_key[(key[0], key[1], "retrieval")]
        reproduction_errors.extend([
            abs(float(actual["candidate_ic"]) - expected["candidate_ic"]),
            abs(float(actual["delta_vs_baseline"]) - expected["delta"]),
        ])
    reproduction_max_error = float(max(reproduction_errors, default=float("inf")))

    summary_lookup = {(row["window"], row["method"]): row for row in summaries}
    direction_metrics: dict[str, Any] = {}
    for direction, left_method, right_method in (
        ("global_minus_retrieval", "global", "retrieval"),
        ("retrieval_minus_global", "retrieval", "global"),
    ):
        pooled, high_drift = [], []
        window_means: dict[str, float] = {}
        random_checks: dict[str, bool] = {}
        for window_name, _, _, _, _ in EVAL_WINDOWS:
            times = sorted(int(row["time"]) for row in per_time_rows if row["window"] == window_name and row["method"] == "retrieval")
            advantage = np.asarray([
                float(by_key[(window_name, time_index, left_method)]["candidate_ic"])
                - float(by_key[(window_name, time_index, right_method)]["candidate_ic"])
                for time_index in times
            ], dtype=np.float64)
            distance = np.asarray([
                float(by_key[(window_name, time_index, "retrieval")]["mean_neighbor_distance"])
                for time_index in times
            ], dtype=np.float64)
            threshold = float(np.quantile(distance, float(params["high_drift_quantile"])))
            pooled.extend(advantage.tolist())
            high_drift.extend(advantage[distance >= threshold].tolist())
            window_means[window_name] = float(np.mean(advantage))
            random_checks[window_name] = bool(
                summary_lookup[(window_name, left_method)]["mean_delta_vs_baseline"] > random_p95[window_name]
            )
        pooled_array = np.asarray(pooled, dtype=np.float64)
        bootstrap = block_bootstrap(
            pooled_array,
            int(params["bootstrap_block_size"]),
            int(params["bootstrap_repetitions"]),
            random_seed + (0 if direction.startswith("global") else 1),
        )
        second_half = pooled_array[pooled_array.size // 2:]
        direction_metrics[direction] = {
            "window_means": window_means,
            "positive_windows": int(sum(value > 0 for value in window_means.values())),
            "fold3_positive": bool(window_means["fold_3"] > 0),
            "official_valid_positive": bool(window_means["official_valid"] > 0),
            "pooled_mean": float(np.mean(pooled_array)),
            "bootstrap": bootstrap,
            "second_half_mean": float(np.mean(second_half)),
            "high_drift_mean": float(np.mean(high_drift)),
            "above_random_p95_by_window": random_checks,
        }

    residual_positive_windows = int(sum(
        summary_lookup[(window_name, "residual")]["mean_delta_vs_baseline"] > 0
        for window_name, _, _, _, _ in EVAL_WINDOWS
    ))

    def common_gate(result: dict[str, Any]) -> dict[str, bool]:
        return {
            "positive_windows": result["positive_windows"] >= int(gates["minimum_positive_windows"]),
            "fold3_positive": result["fold3_positive"],
            "official_valid_positive": result["official_valid_positive"],
            "pooled_gain": result["pooled_mean"] >= float(gates["minimum_pooled_gain"]),
            "bootstrap_lower_positive": result["bootstrap"]["lower_90"] > 0,
            "worst_block": result["bootstrap"]["worst_block"] >= float(gates["minimum_worst_block"]),
            "second_half_nonnegative": result["second_half_mean"] >= 0,
            "high_drift_nonnegative": result["high_drift_mean"] >= 0,
            "above_random_p95_all_windows": all(result["above_random_p95_by_window"].values()),
        }

    global_checks = common_gate(direction_metrics["global_minus_retrieval"])
    state_checks = common_gate(direction_metrics["retrieval_minus_global"])
    state_checks["residual_positive_windows"] = residual_positive_windows >= int(gates["minimum_positive_windows"])
    invariant_checks = {
        "retrieval_reproduction": reproduction_max_error <= float(gates["retrieval_reproduction_tolerance"]),
        "fingerprint_decomposition": decomposition_error < float(gates["decomposition_tolerance"]),
        "causal_neighbor_cutoff": all(
            int(row["max_library_time"]) < int(row["library_cutoff_exclusive"])
            for row in per_time_rows if row["method"] == "retrieval"
        ),
    }
    hashes_after = {name: sha256(path) for name, path in protected.items()}
    invariant_checks["protected_hashes_unchanged"] = hashes_before == hashes_after
    global_pass = all(global_checks.values()) and all(invariant_checks.values())
    state_pass = all(state_checks.values()) and all(invariant_checks.values())
    if global_pass:
        decision = "go_exp027b_global_dominant"
    elif state_pass:
        decision = "go_exp027b_state_specific"
    else:
        decision = "inconclusive_keep_exp024b"

    RESULT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT / "per_time_metrics.csv", per_time_rows)
    write_csv(RESULT / "random_null_summary.csv", random_rows)
    metrics = {
        "experiment": "exp_027a_retrieval_attribution",
        "decision": decision,
        "passed": bool(global_pass or state_pass),
        "features": feature_names,
        "feature_count": len(feature_names),
        "parameters": params,
        "method_summaries": summaries,
        "random_p95_delta_by_window": random_p95,
        "direction_metrics": direction_metrics,
        "residual_positive_windows": residual_positive_windows,
        "global_gate_checks": global_checks,
        "state_gate_checks": state_checks,
        "invariant_checks": invariant_checks,
        "retrieval_reproduction_max_abs_error": reproduction_max_error,
        "fingerprint_decomposition_max_abs_error": decomposition_error,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_unchanged": hashes_before == hashes_after,
        "test_arrays_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "bootstrap_results.json").write_text(json.dumps(direction_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if metrics["passed"] else "completed_rejected",
        "causal_status": "compliant_diagnostic",
        "formal_submission_overwritten": False,
        "test_arrays_loaded": False,
        "test_prediction_generated": False,
        "online_submission_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(invariant_checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
