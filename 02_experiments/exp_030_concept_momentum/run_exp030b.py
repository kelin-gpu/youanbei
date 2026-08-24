from __future__ import annotations

"""exp030b: frozen-history applicability diagnostic for concept-group residual momentum.

Same signal as exp030a, but the group-residual history is frozen at the W sections
immediately before each evaluation window (exactly the deployment shape available on
Test, where labels stop at 3160). fold_1 is skipped because no clean pre-window proxy
history exists. fold2/fold3 select parameters; official valid is a pure holdout with a
second-half horizon gate. No Test labels, no prediction output, no protected writes.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "02_experiments"))
sys.path.insert(0, str(ROOT / "02_experiments" / "exp_024_state_retrieved_rank_residual"))

from run_exp024a import (  # noqa: E402
    COMMON, TREE, VALID_BASE, sha256, safe_corr, rank01, capped_positions, oof_proxy_parts,
)
from run_exp030a import (  # noqa: E402
    section_group_residuals, momentum_signal, worst_block_mean,
)

RESULT = ROOT / "04_results" / "exp_030b_frozen_history_diagnostic"
PROTECTED = {
    "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
    "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
    "exp024b_prediction": ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy",
    "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
}

WINDOWS = (
    ("fold_2", "train", 1945, 2432, "exp016_static_strict_oof_proxy"),
    ("fold_3", "train", 2432, 2918, "exp016_static_strict_oof_proxy"),
    ("official_valid", "valid", 2918, 3161, "exp021_full_valid_frozen"),
)
ALPHA_GRID = (0.05, 0.10, 0.15)
W_GRID = (10, 20, 40)
SEED = 20260824
RANK_CORR_GATE = 0.995
WORST_BLOCK_GATE = -0.002
LOO_BLOCKS = 8
LOO_POSITIVE_RATIO_GATE = 0.75
CAT_SLICE = slice(408, 417)


def main() -> int:
    started = time.time()
    rng = np.random.default_rng(SEED)
    hashes_before = {name: sha256(path) for name, path in PROTECTED.items()}
    RESULT.mkdir(parents=True, exist_ok=True)

    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    valid_groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    proxy = oof_proxy_parts(train_groups)
    valid_base = np.load(VALID_BASE, mmap_mode="r")
    train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
    valid_x = np.load(TREE / "valid_X.npy", mmap_mode="r")
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    train_times_all = np.asarray(np.load(COMMON / "train_time.npy", mmap_mode="r"), dtype=np.int32)
    valid_times_all = np.asarray(np.load(COMMON / "valid_time.npy", mmap_mode="r"), dtype=np.int32)
    train_positions = capped_positions(train_groups)
    train_offsets = np.concatenate([[0], np.cumsum(train_groups)])
    valid_offsets = np.concatenate([[0], np.cumsum(valid_groups)])
    train_section_index = {int(t): i for i, t in enumerate(train_times_all[train_offsets[:-1]])}
    valid_section_index = {int(t): i for i, t in enumerate(valid_times_all[valid_offsets[:-1]])}

    def payload(time_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if time_index < 2918:
            rows_idx = train_positions[train_section_index[time_index]]
            return (np.asarray(train_x[rows_idx][:, CAT_SLICE], dtype=np.float32),
                    np.asarray(train_y[rows_idx], dtype=np.float32),
                    np.asarray(proxy[time_index], dtype=np.float32))
        group_index = valid_section_index[time_index]
        left, right = int(valid_offsets[group_index]), int(valid_offsets[group_index + 1])
        return (np.asarray(valid_x[left:right][:, CAT_SLICE], dtype=np.float32),
                np.asarray(valid_y[left:right], dtype=np.float32),
                np.asarray(valid_base[left:right], dtype=np.float32))

    print("[history] building group residuals for [1459,3161) ...", flush=True)
    history_stack: dict[int, list[dict[float, tuple[float, int]]]] = {}
    payloads: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for time_index in range(1459, 3161):
        cats, y, base = payload(time_index)
        history_stack[time_index] = section_group_residuals(cats, rank01(y) - rank01(base))
        payloads[time_index] = (cats, y, base)
        if (time_index - 1459) % 500 == 0:
            print(f"[history] {time_index}", flush=True)

    records: list[dict[str, object]] = []
    for window_name, split, start, stop, baseline_name in WINDOWS:
        for w in W_GRID:
            snapshot = [history_stack[s] for s in range(start - w, start)]
            for time_index in range(start, stop):
                cats, y, base = payloads[time_index]
                base_rank = rank01(base)
                y_rank = rank01(y)
                baseline_ic = safe_corr(base_rank, y_rank)
                signal = momentum_signal(cats, snapshot)
                shuffled = signal[rng.permutation(signal.size)]
                for values, method in ((signal, "momentum"), (shuffled, "shuffled")):
                    ranked = rank01(values)
                    for alpha in ALPHA_GRID:
                        candidate = rank01((1.0 - alpha) * base_rank + alpha * ranked)
                        candidate_ic = safe_corr(candidate, y_rank)
                        records.append({
                            "window": window_name, "time": time_index, "baseline": baseline_name,
                            "method": method, "alpha": alpha, "history_w": w,
                            "distance_from_snapshot": time_index - start,
                            "baseline_ic": baseline_ic, "candidate_ic": candidate_ic,
                            "delta": candidate_ic - baseline_ic,
                            "rank_correlation_to_baseline": safe_corr(base_rank, candidate),
                        })
        print(f"[{window_name}] done", flush=True)

    with (RESULT / "per_time_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def window_stats(window: str, method: str, alpha: float, w: int) -> dict[str, float]:
        chosen = [r for r in records if r["window"] == window and r["method"] == method
                  and r["alpha"] == alpha and r["history_w"] == w]
        deltas = np.asarray([r["delta"] for r in chosen], dtype=np.float64)
        corrs = np.asarray([r["rank_correlation_to_baseline"] for r in chosen], dtype=np.float64)
        return {
            "sections": float(deltas.size),
            "baseline_mean_ic": float(np.mean([r["baseline_ic"] for r in chosen])),
            "candidate_mean_ic": float(np.mean([r["candidate_ic"] for r in chosen])),
            "mean_delta": float(np.mean(deltas)),
            "positive_ratio": float(np.mean(deltas > 0)),
            "min_rank_correlation": float(np.min(corrs)),
            "worst_block_mean": worst_block_mean(deltas),
        }

    selection: list[dict[str, object]] = []
    eligible: list[tuple[float, float, int]] = []
    for alpha in ALPHA_GRID:
        for w in W_GRID:
            f2 = window_stats("fold_2", "momentum", alpha, w)
            f3 = window_stats("fold_3", "momentum", alpha, w)
            s2 = window_stats("fold_2", "shuffled", alpha, w)
            s3 = window_stats("fold_3", "shuffled", alpha, w)
            pooled = 0.5 * (f2["mean_delta"] + f3["mean_delta"])
            shuffled_pooled = 0.5 * (s2["mean_delta"] + s3["mean_delta"])
            constraints = (
                f2["mean_delta"] > 0 and f3["mean_delta"] > 0
                and f2["min_rank_correlation"] >= RANK_CORR_GATE
                and f3["min_rank_correlation"] >= RANK_CORR_GATE
                and f2["worst_block_mean"] >= WORST_BLOCK_GATE
                and f3["worst_block_mean"] >= WORST_BLOCK_GATE
                and pooled > shuffled_pooled
            )
            selection.append({
                "alpha": alpha, "history_w": w, "pooled_fold2_fold3_delta": pooled,
                "shuffled_pooled_fold2_fold3_delta": shuffled_pooled,
                "fold2": f2, "fold3": f3, "constraints_passed": bool(constraints),
            })
            if constraints:
                eligible.append((pooled, -alpha, w))

    pass_checks: dict[str, object] = {"any_combo_eligible_on_folds": bool(bool(eligible))}
    chosen_alpha, chosen_w, chosen_pooled = None, None, None
    valid_stats = shuffled_valid_stats = None
    decay: list[dict[str, object]] = []
    extra: dict[str, object] = {}
    if eligible:
        eligible.sort(reverse=True)
        chosen_pooled, neg_alpha, chosen_w = eligible[0]
        chosen_alpha = -neg_alpha
        valid_stats = window_stats("official_valid", "momentum", chosen_alpha, chosen_w)
        shuffled_valid_stats = window_stats("official_valid", "shuffled", chosen_alpha, chosen_w)
        valid_rows = [r for r in records if r["window"] == "official_valid" and r["method"] == "momentum"
                      and r["alpha"] == chosen_alpha and r["history_w"] == chosen_w]
        valid_rows.sort(key=lambda r: int(r["time"]))
        valid_deltas = np.asarray([r["delta"] for r in valid_rows], dtype=np.float64)
        half = valid_deltas.size // 2
        second_half_delta = float(np.mean(valid_deltas[half:]))
        for piece in np.array_split(valid_deltas, 4):
            decay.append({"quartile_mean_delta": float(np.mean(piece)), "sections": int(piece.size)})
        chosen_deltas = np.asarray([
            r["delta"] for r in records
            if r["method"] == "momentum" and r["alpha"] == chosen_alpha and r["history_w"] == chosen_w
        ], dtype=np.float64)
        loo_means = []
        for piece in np.array_split(np.arange(chosen_deltas.size), min(LOO_BLOCKS, chosen_deltas.size)):
            keep = np.ones(chosen_deltas.size, dtype=bool)
            keep[piece] = False
            loo_means.append(float(np.mean(chosen_deltas[keep])))
        loo_positive_ratio = float(np.mean(np.asarray(loo_means) > 0))
        pass_checks.update({
            "momentum_beats_shuffled_on_fold2_fold3_pooled": True,
            "official_valid_pooled_delta_nonnegative": bool(valid_stats["mean_delta"] >= 0.0),
            "official_valid_second_half_delta_nonnegative": bool(second_half_delta >= 0.0),
            "official_valid_worst_block_at_least_-0_002": bool(valid_stats["worst_block_mean"] >= WORST_BLOCK_GATE),
            "official_valid_rank_correlation_min_0_995": bool(valid_stats["min_rank_correlation"] >= RANK_CORR_GATE),
            "loo_delta_positive_ratio_at_least_0_75": bool(loo_positive_ratio >= LOO_POSITIVE_RATIO_GATE),
            "protected_hashes_unchanged": bool(hashes_before == {n: sha256(p) for n, p in PROTECTED.items()}),
        })
        extra = {"loo_positive_ratio": loo_positive_ratio, "valid_second_half_delta": second_half_delta}

    passed = bool(all(pass_checks.values()))
    (RESULT / "protocol.json").write_text(json.dumps({
        "experiment": "exp_030b_frozen_history_diagnostic",
        "frozen_parameters": {
            "alpha_grid": list(ALPHA_GRID), "history_window_grid_sections": list(W_GRID),
            "selection_rule": "max pooled fold2+fold3 subject to fold gates and momentum>shuffled",
            "chosen_alpha": chosen_alpha, "chosen_history_w": chosen_w,
            "pooled_fold2_fold3_delta": chosen_pooled,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "experiment": "exp_030b_frozen_history_diagnostic",
        "decision": "frozen_deployment_supported_generate_exp030c" if passed else "frozen_deployment_unsupported_stop",
        "passed": passed,
        "selection_grid": selection,
        "chosen_parameters": {"alpha": chosen_alpha, "history_w": chosen_w,
                              "pooled_fold2_fold3_delta": chosen_pooled},
        "official_valid_stats": valid_stats,
        "official_valid_shuffled_stats": shuffled_valid_stats,
        "valid_distance_decay_quartiles": decay,
        "pass_checks": pass_checks,
        "extra": extra,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": {name: sha256(path) for name, path in PROTECTED.items()},
        "test_labels_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if passed else "completed_rejected",
        "causal_status": "compliant_diagnostic_no_test_labels",
        "baseline_note": "frozen snapshots: fold2=[1935,1945), fold3=[2422,2432), valid=[2908,2918); fold1 skipped (no pre-window proxy history)",
        "formal_submission_overwritten": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in
                      ("decision", "passed", "chosen_parameters", "pass_checks", "extra",
                       "valid_distance_decay_quartiles")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
