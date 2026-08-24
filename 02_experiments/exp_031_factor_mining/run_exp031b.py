from __future__ import annotations

"""exp031b: user-authorized exploratory candidate.

Replicates run_exp024b.py exactly (frozen exp024a protocol parameters) with the only
difference being the retrieval container's feature contract extended from the 31
registry-allowed base columns to 31 + 4 mined rank-diff factors, using the exact
same factor_lib.compute_factor implementation validated in exp031a. Test labels are
never loaded; protected files are hash-checked before and after; no auto submission.
"""

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
sys.path.insert(0, str(ROOT / "02_experiments" / "exp_024_state_retrieved_rank_residual"))

from run_exp024a import (  # noqa: E402
    COMMON, TREE, correction_score, feature_contract,
    retrieval_coordinates, section_state, section_fingerprint, sha256,
)
from factor_lib import factor_name, extended_features  # noqa: E402
from exp_016_unified_expert_fusion.src.prediction_contract import validate_prediction  # noqa: E402

BASE_PATH = ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy"
EXP024B_PREDICTION = ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy"
RESULT = ROOT / "04_results" / "exp_031b_factor_mined_candidate"
PROTOCOL_024B = ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "protocol.json"
PROTOCOL_031A = ROOT / "04_results" / "exp_031a_factor_mining_diagnostic" / "protocol.json"
TEST_START, TEST_STOP = 3161, 3603

PROTECTED = {
    "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
    "exp023h_prediction": BASE_PATH,
    "exp024b_prediction": EXP024B_PREDICTION,
    "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
}


def rank01(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(values, method="average")
    return ((ranked - 1.0) / max(1, ranked.size - 1)).astype(np.float32)


def build_split_panel(split: str, base_positions: list[int],
                      factor_specs: list[dict]) -> dict[str, np.ndarray]:
    x_all = np.load(TREE / f"{split}_X.npy", mmap_mode="r")
    y_all = np.load(COMMON / f"{split}_y.npy", mmap_mode="r")
    times_all = np.load(COMMON / f"{split}_time.npy", mmap_mode="r")
    groups = np.asarray(np.load(COMMON / f"{split}_group_sizes.npy"), dtype=np.int64)
    states, fingerprints, times = [], [], []
    offset = 0
    for group_index, size_value in enumerate(groups):
        size = int(size_value)
        block = np.asarray(x_all[offset:offset + size], dtype=np.float32)
        extended = extended_features(block, base_positions, factor_specs)
        y = np.asarray(y_all[offset:offset + size], dtype=np.float32)
        states.append(section_state(extended))
        fingerprints.append(section_fingerprint(extended, y))
        times.append(int(times_all[offset]))
        offset += size
        if (group_index + 1) % 300 == 0:
            print(f"[{split}] summarized {group_index + 1}/{groups.size} sections", flush=True)
    return {
        "states": np.stack(states),
        "fingerprints": np.stack(fingerprints),
        "times": np.asarray(times, dtype=np.int32),
        "groups": groups,
    }


def build_test_panel(base_positions: list[int], factor_specs: list[dict]):
    x_all = np.load(TREE / "test_X.npy", mmap_mode="r")
    groups = np.asarray(np.load(COMMON / "test_group_sizes.npy"), dtype=np.int64)
    states = []
    offset = 0
    for group_index, size_value in enumerate(groups):
        size = int(size_value)
        block = np.asarray(x_all[offset:offset + size], dtype=np.float32)
        extended = extended_features(block, base_positions, factor_specs)
        states.append(section_state(extended))
        offset += size
        if (group_index + 1) % 100 == 0:
            print(f"[test] summarized {group_index + 1}/{groups.size} sections", flush=True)
    return np.stack(states), groups


def main() -> int:
    started = time.time()
    params = json.loads(PROTOCOL_024B.read_text(encoding="utf-8"))["preregistered_parameters"]
    k = int(params["neighbors"])
    alpha = float(params["candidate_shrink_alpha"])
    preserve = int(params["preserve_exp023h_first_sections"])
    pca_components = int(params["state_pca_components"])
    factor_specs = json.loads(PROTOCOL_031A.read_text(encoding="utf-8"))["final_factor_specs"]
    factor_names = [factor_name(spec) for spec in factor_specs]
    print(f"frozen params: K={k} alpha={alpha} preserve={preserve} pca={pca_components}", flush=True)
    print(f"factors: {factor_names}", flush=True)
    hashes_before = {name: sha256(path) for name, path in PROTECTED.items()}

    base_positions, base_names = feature_contract()
    print(f"base features={len(base_positions)} + factors={len(factor_specs)}", flush=True)
    train = build_split_panel("train", base_positions, factor_specs)
    valid = build_split_panel("valid", base_positions, factor_specs)
    test_states, test_groups = build_test_panel(base_positions, factor_specs)
    library_states = np.concatenate([train["states"], valid["states"]])
    library_fingerprints = np.concatenate([train["fingerprints"], valid["fingerprints"]])
    library_times = np.concatenate([train["times"], valid["times"]])
    library_coord, test_coord = retrieval_coordinates(library_states, test_states, pca_components)

    test_tree = np.load(TREE / "test_X.npy", mmap_mode="r")
    test_times = np.asarray(np.load(COMMON / "test_time.npy", mmap_mode="r"), dtype=np.int32)
    test_stocks = np.asarray(np.load(COMMON / "test_stock.npy", mmap_mode="r"), dtype=np.int32)
    base_grid = np.asarray(np.load(BASE_PATH), dtype=np.float32)
    exp024b_grid = np.asarray(np.load(EXP024B_PREDICTION), dtype=np.float32)
    output = base_grid.copy()
    offsets = np.concatenate([[0], np.cumsum(test_groups)])
    neighbor_rows = []
    corr_to_base_list, corr_to_024b_list = [], []
    changed_sections = 0

    for group_index, size_value in enumerate(test_groups):
        left, right = int(offsets[group_index]), int(offsets[group_index + 1])
        time_index = int(test_times[left])
        stocks = test_stocks[left:right]
        base = base_grid[time_index - TEST_START, stocks]
        exp024b_values = exp024b_grid[time_index - TEST_START, stocks]
        candidate = base.copy()
        mean_distance = float("nan")
        picked_times: np.ndarray = np.empty(0, dtype=np.int32)
        if group_index >= preserve:
            distances = np.sqrt(np.sum((library_coord - test_coord[group_index]) ** 2, axis=1))
            nearest = np.argpartition(distances, k - 1)[:k]
            fingerprint = np.mean(library_fingerprints[nearest], axis=0)
            block = np.asarray(test_tree[left:right], dtype=np.float32)
            extended = extended_features(block, base_positions, factor_specs)
            correction = correction_score(extended, fingerprint)
            candidate = (1.0 - alpha) * rank01(base) + alpha * rank01(correction)
            candidate = rank01(candidate)
            mean_distance = float(np.mean(distances[nearest]))
            picked_times = library_times[nearest]
        output[time_index - TEST_START, stocks] = candidate.astype(np.float32)
        corr_to_base = float(np.corrcoef(rankdata(base), rankdata(candidate))[0, 1])
        corr_to_024b = float(np.corrcoef(rankdata(exp024b_values), rankdata(candidate))[0, 1])
        corr_to_base_list.append(corr_to_base)
        corr_to_024b_list.append(corr_to_024b)
        if not np.array_equal(base.astype(np.float32), candidate.astype(np.float32)):
            changed_sections += 1
        neighbor_rows.append({
            "test_time": time_index,
            "preserved": group_index < preserve,
            "mean_neighbor_distance": mean_distance,
            "min_library_time": int(np.min(picked_times)) if picked_times.size else "missing",
            "max_library_time": int(np.max(picked_times)) if picked_times.size else "missing",
            "rank_correlation_to_exp023h": corr_to_base,
            "rank_correlation_to_exp024b": corr_to_024b,
        })
        if (group_index + 1) % 100 == 0:
            print(f"[test] corrected {group_index + 1}/{test_groups.size} sections", flush=True)

    mask = np.zeros_like(output, dtype=bool)
    mask[test_times - TEST_START, test_stocks] = True
    contract = validate_prediction(output, mask)
    min_corr_to_024b = float(np.min(corr_to_024b_list[preserve:])) if len(corr_to_024b_list) > preserve else 1.0
    hashes_after = {name: sha256(path) for name, path in PROTECTED.items()}
    delivery_ok = bool(contract["finite"] and min_corr_to_024b >= 0.995
                       and hashes_before == hashes_after)

    if delivery_ok:
        RESULT.mkdir(parents=True, exist_ok=True)
        np.save(RESULT / "prediction_1.npy", output)
        with (RESULT / "retrieval_neighbors.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(neighbor_rows[0]))
            writer.writeheader()
            writer.writerows(neighbor_rows)
    else:
        print("DELIVERY GATE FAILED — prediction not saved", flush=True)

    metrics = {
        "experiment": "exp_031b_factor_mined_candidate",
        "decision": "exploratory_candidate_generated_not_promoted" if delivery_ok else "delivery_gate_failed",
        "authorization": "user_authorized_exploratory_candidate_2026-08-24",
        "baseline": "exp023h_ultimate_surgery (container identical to exp024b)",
        "difference_from_exp024b": "feature_contract extended 31 -> 35 (4 mined rank-diff factors from exp031a)",
        "frozen_parameters": {key: params[key] for key in
                              ("neighbors", "candidate_shrink_alpha",
                               "preserve_exp023h_first_sections", "state_pca_components")},
        "factor_specs": factor_specs,
        "factor_names": factor_names,
        "features": base_names + factor_names,
        "prediction": "prediction_1.npy" if delivery_ok else None,
        "prediction_sha256": sha256(RESULT / "prediction_1.npy") if delivery_ok else None,
        "contract": contract,
        "preserved_first_sections": preserve,
        "changed_sections_vs_exp023h": changed_sections,
        "per_time_rank_correlation_to_exp023h": {
            "mean": float(np.mean(corr_to_base_list)),
            "min": float(np.min(corr_to_base_list)),
        },
        "per_time_rank_correlation_to_exp024b": {
            "mean": float(np.mean(corr_to_024b_list)),
            "min_all_sections": float(np.min(corr_to_024b_list)),
            "min_corrected_sections": min_corr_to_024b,
            "gate": 0.995,
        },
        "delivery_gate": {
            "contract_finite": bool(contract["finite"]),
            "min_rank_correlation_to_exp024b": min_corr_to_024b,
            "protected_unchanged": hashes_before == hashes_after,
            "passed": delivery_ok,
        },
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "test_labels_loaded": False,
        "automatic_online_submission": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "candidate_generated_user_authorized_not_promoted" if delivery_ok else "delivery_gate_failed",
        "causal_status": "compliant_claim_and_contract_recorded",
        "formal_submission_overwritten": False,
        "baseline_overwritten": False,
        "warning": "exp031a failed gate g6 (a pre-existing container property); submit only as a user-authorized exploratory probe, mirroring the exp024b precedent",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in
                      ("decision", "delivery_gate", "per_time_rank_correlation_to_exp024b",
                       "changed_sections_vs_exp023h")}, ensure_ascii=False, indent=2), flush=True)
    return 0 if delivery_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
