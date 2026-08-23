from __future__ import annotations

"""Check temporal transfer of exact-Spearman optimized family weights.

No Test data or prediction path is opened.  Existing strict-OOF family outputs
are used for two expanding holdouts; frozen exp021 is used only as the official
Valid baseline.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_experiments"))

from exp_016_unified_expert_fusion.config import BASE_WEIGHTS, FAMILIES  # noqa: E402


COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
OOF_PATH = ROOT / "03_cache" / "exp_016_unified_expert_fusion" / "oof_predictions" / "family_matrix.npy"
VALID_FAMILY_PATH = ROOT / "04_results" / "exp_016_unified_expert_fusion" / "full" / "full_valid_family_predictions.npy"
VALID_BASE_PATH = ROOT / "04_results" / "_acceptance" / "exp021_validation_audit" / "full_valid_prediction.npy"
RESULT = ROOT / "04_results" / "exp_025a_direct_rank_objective_diagnostic"
PROTOCOL = RESULT / "protocol.json"

OOF_START = 1459
FOLD_1_STOP = 1945
FOLD_2_STOP = 2432
OOF_STOP = 2918


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_ranks(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, axis=1, method="average")
    return np.asarray(ranked, dtype=np.float32)


def row_rank_ic(scores: np.ndarray, target_ranks: np.ndarray) -> np.ndarray:
    ranked = row_ranks(scores).astype(np.float64)
    target = np.asarray(target_ranks, dtype=np.float64)
    ranked -= ranked.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(ranked * ranked, axis=1) * np.sum(target * target, axis=1))
    return np.divide(np.sum(ranked * target, axis=1), denom,
                     out=np.zeros(ranked.shape[0], dtype=np.float64), where=denom > 1e-12)


def oof_targets() -> np.ndarray:
    groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    offsets = np.concatenate([[0], np.cumsum(groups)])
    parts = []
    for time_index in range(OOF_START, OOF_STOP):
        group_index = time_index - 486
        left, right = int(offsets[group_index]), int(offsets[group_index + 1])
        size = right - left
        selected = left + np.linspace(0, size - 1, min(size, 1024), dtype=np.int64)
        if selected.size != 1024:
            raise RuntimeError("Expected 1024 capped OOF rows per time section")
        parts.append(np.asarray(y[selected], dtype=np.float32))
    return np.stack(parts)


def exact_panel(groups: np.ndarray, values: np.ndarray, target: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    value_parts, target_parts = [], []
    offset = 0
    for size_value in groups:
        size = int(size_value)
        value_parts.append(np.asarray(values[offset:offset + size], dtype=np.float32))
        target_parts.append(np.asarray(target[offset:offset + size], dtype=np.float32))
        offset += size
    return value_parts, target_parts


def group_rank_matrix(parts: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(rankdata(part, axis=0, method="average"), dtype=np.float32) for part in parts]


def fit_weights(family_ranks: np.ndarray, target_ranks: np.ndarray, base_weights: np.ndarray,
                alpha: float, penalty: float, maxfev: int) -> dict[str, object]:
    base_score = np.tensordot(family_ranks, base_weights, axes=([2], [0]))
    initial = np.log(np.clip(base_weights, 1e-6, None))
    calls = 0

    def objective(logits: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        weights = softmax(logits)
        direction = np.tensordot(family_ranks, weights, axes=([2], [0]))
        candidate = (1.0 - alpha) * base_score + alpha * direction
        mean_ic = float(np.mean(row_rank_ic(candidate, target_ranks)))
        return -mean_ic + penalty * float(np.sum((weights - base_weights) ** 2))

    fitted = minimize(objective, initial, method="Nelder-Mead",
                      options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-6, "disp": False})
    weights = softmax(fitted.x)
    return {
        "weights": weights,
        "success": bool(fitted.success),
        "message": str(fitted.message),
        "function_evaluations": int(fitted.nfev),
        "objective_calls": calls,
        "train_penalized_objective": float(fitted.fun),
    }


def evaluate_oof(family_ranks: np.ndarray, target_ranks: np.ndarray, base_weights: np.ndarray,
                 fitted_weights: np.ndarray, alpha: float, start: int, stop: int) -> dict[str, float]:
    family = family_ranks[start:stop]
    target = target_ranks[start:stop]
    base_score = np.tensordot(family, base_weights, axes=([2], [0]))
    direction = np.tensordot(family, fitted_weights, axes=([2], [0]))
    candidate = (1.0 - alpha) * base_score + alpha * direction
    base_ic = row_rank_ic(base_score, target)
    candidate_ic = row_rank_ic(candidate, target)
    delta = candidate_ic - base_ic
    return {
        "sections": int(stop - start),
        "baseline_mean_ic": float(np.mean(base_ic)),
        "candidate_mean_ic": float(np.mean(candidate_ic)),
        "mean_delta": float(np.mean(delta)),
        "positive_delta_ratio": float(np.mean(delta > 0)),
        "worst_delta": float(np.min(delta)),
    }


def evaluate_valid(fitted_weights: np.ndarray, base_weights: np.ndarray, alpha: float) -> dict[str, float]:
    groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    target = np.asarray(np.load(COMMON / "valid_y.npy", mmap_mode="r"), dtype=np.float32)
    family = np.asarray(np.load(VALID_FAMILY_PATH, mmap_mode="r"), dtype=np.float32)
    frozen_base = np.asarray(np.load(VALID_BASE_PATH, mmap_mode="r"), dtype=np.float32)
    family_parts, target_parts = exact_panel(groups, family, target)
    family_rank_parts = group_rank_matrix(family_parts)
    base_parts, _ = exact_panel(groups, frozen_base[:, None], target)
    base_ics, candidate_ics = [], []
    for fam_rank, base_part, target_part in zip(family_rank_parts, base_parts, target_parts):
        direction = np.asarray(fam_rank @ fitted_weights, dtype=np.float32)
        base_rank = np.asarray(rankdata(base_part[:, 0], method="average"), dtype=np.float32)
        candidate = (1.0 - alpha) * base_rank + alpha * direction
        target_rank = np.asarray(rankdata(target_part, method="average"), dtype=np.float32)
        base_ics.append(float(np.corrcoef(rankdata(base_rank), target_rank)[0, 1]))
        candidate_ics.append(float(np.corrcoef(rankdata(candidate), target_rank)[0, 1]))
    base_arr = np.asarray(base_ics)
    candidate_arr = np.asarray(candidate_ics)
    delta = candidate_arr - base_arr
    return {
        "sections": int(groups.size),
        "baseline_mean_ic": float(np.mean(base_arr)),
        "candidate_mean_ic": float(np.mean(candidate_arr)),
        "mean_delta": float(np.mean(delta)),
        "positive_delta_ratio": float(np.mean(delta > 0)),
        "worst_delta": float(np.min(delta)),
    }


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    params = protocol["preregistered_parameters"]
    alpha = float(params["candidate_shrink_alpha"])
    penalty = float(params["l2_penalty_to_base_weights"])
    maxfev = int(params["optimizer_maxfev"])
    base_weights = np.asarray([BASE_WEIGHTS[name] for name in FAMILIES], dtype=np.float64)
    base_weights /= base_weights.sum()

    protected = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    hashes_before = {name: sha256(path) for name, path in protected.items()}

    matrix = np.asarray(np.load(OOF_PATH, mmap_mode="r"), dtype=np.float32)
    expected_sections = OOF_STOP - OOF_START
    if matrix.shape != (expected_sections * 1024, len(FAMILIES)):
        raise RuntimeError(f"Unexpected OOF matrix shape: {matrix.shape}")
    family = matrix.reshape(expected_sections, 1024, len(FAMILIES))
    print("ranking strict-OOF family matrix", flush=True)
    family_ranks = row_ranks(family)
    targets = oof_targets()
    target_ranks = row_ranks(targets)

    fold1_end = FOLD_1_STOP - OOF_START
    fold2_end = FOLD_2_STOP - OOF_START
    print("fit fold_1 -> evaluate fold_2", flush=True)
    fit_1 = fit_weights(family_ranks[:fold1_end], target_ranks[:fold1_end], base_weights, alpha, penalty, maxfev)
    eval_2 = evaluate_oof(family_ranks, target_ranks, base_weights, fit_1["weights"], alpha, fold1_end, fold2_end)
    print("fit folds_1_2 -> evaluate fold_3", flush=True)
    fit_2 = fit_weights(family_ranks[:fold2_end], target_ranks[:fold2_end], base_weights, alpha, penalty, maxfev)
    eval_3 = evaluate_oof(family_ranks, target_ranks, base_weights, fit_2["weights"], alpha, fold2_end, expected_sections)
    print("fit all OOF -> evaluate official Valid", flush=True)
    fit_all = fit_weights(family_ranks, target_ranks, base_weights, alpha, penalty, maxfev)
    eval_valid = evaluate_valid(fit_all["weights"], base_weights, alpha)

    instability = float(np.sum(np.abs(fit_1["weights"] - fit_2["weights"])))
    hashes_after = {name: sha256(path) for name, path in protected.items()}
    pass_checks = {
        "fold_2_delta_positive": eval_2["mean_delta"] > 0,
        "fold_3_delta_positive": eval_3["mean_delta"] > 0,
        "official_valid_delta_at_least_0_0025": eval_valid["mean_delta"] >= 0.0025,
        "weight_l1_instability_at_most_0_20": instability <= float(params["weight_stability_l1_max"]),
        "protected_hashes_unchanged": hashes_before == hashes_after,
    }
    passed = bool(all(pass_checks.values()))

    def serial_fit(value: dict[str, object]) -> dict[str, object]:
        copied = dict(value)
        copied["weights"] = {name: float(weight) for name, weight in zip(FAMILIES, value["weights"])}
        return copied

    metrics = {
        "experiment": "exp_025a_direct_rank_objective_diagnostic",
        "decision": "go_to_soft_rank_calibrator" if passed else "stop_direct_rank_route",
        "passed": passed,
        "parameters": params,
        "base_weights": {name: float(weight) for name, weight in zip(FAMILIES, base_weights)},
        "fit_fold_1": serial_fit(fit_1),
        "evaluation_fold_2": eval_2,
        "fit_folds_1_2": serial_fit(fit_2),
        "evaluation_fold_3": eval_3,
        "fit_all_oof": serial_fit(fit_all),
        "evaluation_official_valid": eval_valid,
        "fit_weight_l1_instability": instability,
        "pass_checks": pass_checks,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_unchanged": hashes_before == hashes_after,
        "test_arrays_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if passed else "completed_rejected",
        "causal_status": "compliant_strict_oof_diagnostic",
        "formal_submission_overwritten": False,
        "test_prediction_generated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
