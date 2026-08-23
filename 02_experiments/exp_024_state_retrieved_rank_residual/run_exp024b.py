from __future__ import annotations

"""Generate the fixed, user-requested exploratory retrieval candidate."""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "02_experiments"))

from run_exp024a import (  # noqa: E402
    COMMON, TREE, build_panel, correction_score, feature_contract,
    retrieval_coordinates, section_state, sha256,
)
from exp_016_unified_expert_fusion.src.prediction_contract import validate_prediction  # noqa: E402


BASE_PATH = ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy"
RESULT = ROOT / "04_results" / "exp_024b_retrieval_exploratory"
PROTOCOL = RESULT / "protocol.json"
TEST_START, TEST_STOP = 3161, 3603


def rank01(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    return ((ranked - 1.0) / max(1, ranked.size - 1)).astype(np.float32)


def build_test_states(feature_positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(np.load(COMMON / "test_group_sizes.npy"), dtype=np.int64)
    tree = np.load(TREE / "test_X.npy", mmap_mode="r")
    states = []
    offset = 0
    for group_index, size_value in enumerate(groups):
        size = int(size_value)
        x = np.asarray(tree[offset:offset + size, feature_positions], dtype=np.float32)
        states.append(section_state(x))
        offset += size
        if (group_index + 1) % 100 == 0:
            print(f"[test] summarized {group_index + 1}/{groups.size} sections", flush=True)
    return np.stack(states), groups


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    params = protocol["preregistered_parameters"]
    k = int(params["neighbors"])
    alpha = float(params["candidate_shrink_alpha"])
    preserve = int(params["preserve_exp023h_first_sections"])
    protected = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "exp023h_prediction": BASE_PATH,
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    hashes_before = {name: sha256(path) for name, path in protected.items()}

    feature_positions, feature_names = feature_contract()
    print(f"eligible features={len(feature_positions)}", flush=True)
    train = build_panel("train", feature_positions)
    valid = build_panel("valid", feature_positions)
    test_states, test_groups = build_test_states(feature_positions)
    library_states = np.concatenate([train["states"], valid["states"]])
    library_fingerprints = np.concatenate([train["fingerprints"], valid["fingerprints"]])
    library_times = np.concatenate([train["times"], valid["times"]])
    library_coord, test_coord = retrieval_coordinates(
        library_states, test_states, int(params["state_pca_components"])
    )

    test_tree = np.load(TREE / "test_X.npy", mmap_mode="r")
    test_times = np.asarray(np.load(COMMON / "test_time.npy", mmap_mode="r"), dtype=np.int32)
    test_stocks = np.asarray(np.load(COMMON / "test_stock.npy", mmap_mode="r"), dtype=np.int32)
    base_grid = np.asarray(np.load(BASE_PATH), dtype=np.float32)
    output = base_grid.copy()
    offsets = np.concatenate([[0], np.cumsum(test_groups)])
    neighbor_rows = []
    per_time_correlations = []
    changed_sections = 0

    for group_index, size_value in enumerate(test_groups):
        left, right = int(offsets[group_index]), int(offsets[group_index + 1])
        size = int(size_value)
        time_index = int(test_times[left])
        stocks = test_stocks[left:right]
        base = base_grid[time_index - TEST_START, stocks]
        candidate = base.copy()
        mean_distance = float("nan")
        picked_times: np.ndarray = np.empty(0, dtype=np.int32)
        if group_index >= preserve:
            distances = np.sqrt(np.sum((library_coord - test_coord[group_index]) ** 2, axis=1))
            nearest = np.argpartition(distances, k - 1)[:k]
            fingerprint = np.mean(library_fingerprints[nearest], axis=0)
            x = np.asarray(test_tree[left:right, feature_positions], dtype=np.float32)
            correction = correction_score(x, fingerprint)
            candidate = (1.0 - alpha) * rank01(base) + alpha * rank01(correction)
            candidate = rank01(candidate)
            mean_distance = float(np.mean(distances[nearest]))
            picked_times = library_times[nearest]
        output[time_index - TEST_START, stocks] = candidate.astype(np.float32)
        correlation = float(np.corrcoef(rankdata(base), rankdata(candidate))[0, 1])
        per_time_correlations.append(correlation)
        if not np.array_equal(base.astype(np.float32), candidate.astype(np.float32)):
            changed_sections += 1
        neighbor_rows.append({
            "test_time": time_index,
            "preserved": group_index < preserve,
            "mean_neighbor_distance": mean_distance,
            "min_library_time": int(np.min(picked_times)) if picked_times.size else "missing",
            "max_library_time": int(np.max(picked_times)) if picked_times.size else "missing",
            "rank_correlation_to_exp023h": correlation,
        })
        if (group_index + 1) % 100 == 0:
            print(f"[test] corrected {group_index + 1}/{test_groups.size} sections", flush=True)

    mask = np.zeros_like(output, dtype=bool)
    mask[test_times - TEST_START, test_stocks] = True
    contract = validate_prediction(output, mask)
    RESULT.mkdir(parents=True, exist_ok=True)
    np.save(RESULT / "prediction_1.npy", output)
    with (RESULT / "retrieval_neighbors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(neighbor_rows[0]))
        writer.writeheader()
        writer.writerows(neighbor_rows)

    hashes_after = {name: sha256(path) for name, path in protected.items()}
    candidate_hash = sha256(RESULT / "prediction_1.npy")
    evaluation_base = base_grid[mask]
    evaluation_candidate = output[mask]
    metrics = {
        "experiment": "exp_024b_retrieval_exploratory",
        "decision": "exploratory_candidate_generated_not_promoted",
        "baseline": "exp023h_ultimate_surgery",
        "parameters": params,
        "feature_count": len(feature_names),
        "features": feature_names,
        "prediction": "prediction_1.npy",
        "prediction_sha256": candidate_hash,
        "contract": contract,
        "preserved_first_sections": preserve,
        "changed_sections": changed_sections,
        "evaluation_pearson_to_exp023h": float(np.corrcoef(evaluation_base, evaluation_candidate)[0, 1]),
        "per_time_rank_correlation_mean": float(np.mean(per_time_correlations)),
        "per_time_rank_correlation_min": float(np.min(per_time_correlations)),
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_unchanged": hashes_before == hashes_after,
        "test_labels_loaded": False,
        "automatic_online_submission": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "candidate_generated_exploratory_not_promoted",
        "causal_status": "compliant_claim_and_contract_recorded",
        "formal_submission_overwritten": False,
        "baseline_overwritten": False,
        "warning": "exp024a failed robustness gates; submit only as a user-authorized exploratory probe",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0 if contract["finite"] and hashes_before == hashes_after else 1


if __name__ == "__main__":
    raise SystemExit(main())
