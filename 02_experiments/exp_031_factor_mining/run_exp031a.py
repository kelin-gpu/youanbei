from __future__ import annotations

"""exp031a: deterministic causal factor mining into the exp024b retrieval container.

Stage A screens cross-sectional factor templates (rank/diff/prod/concept-relative/
abs-deviation over the 31 registry-allowed features) by pooled fold2+fold3 IC with
sign consistency. Stage B computes full-section state/fingerprint columns for the
top screened candidates. Stage C greedily extends the container's feature contract,
scoring each subset by the frozen exp024a retrieval protocol (PCA16, K=32, alpha=0.1)
on fold2+fold3; official valid is a pure holdout. No Test labels, no prediction
output, no protected-file writes.
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
    COMMON, TREE, VALID_BASE, EVAL_WINDOWS, sha256, safe_corr, rank01, rank_columns,
    feature_contract, build_panel, section_state, section_fingerprint,
    capped_positions, oof_proxy_parts, retrieval_coordinates, correction_score,
)
from factor_lib import compute_factor, factor_name, group_demean, extended_features  # noqa: E402

RESULT = ROOT / "04_results" / "exp_031a_factor_mining_diagnostic"
EXP024A_METRICS = ROOT / "04_results" / "exp_024a_retrieval_diagnostic" / "metrics.json"
PROTECTED = {
    "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
    "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
    "exp024b_prediction": ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy",
    "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
}

FOLD2, FOLD3 = (1945, 2432), (2432, 2918)
TOP_T = 16
IC_FLOOR = 0.005
GREEDY_TRY = 12
GREEDY_KEEP_MAX = 4
GREEDY_STOP_AFTER_FAILS = 3
PCA_COMPONENTS = 16
K = 32
ALPHA = 0.1
PCA_SEED = 20260823
CONTROL_SEED = 20260824
WORST_BLOCK_GATE = -0.002
LOO_BLOCKS = 8
LOO_RATIO_GATE = 0.75


def col_corr(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    left = matrix - matrix.mean(axis=0, keepdims=True)
    right = target - target.mean()
    num = np.sum(left * right[:, None], axis=0)
    den = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)


def screening() -> tuple[list[dict[str, object]], list[int], np.ndarray]:
    base_positions, _ = feature_contract()
    n_base = len(base_positions)
    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    train_times = np.asarray(np.load(COMMON / "train_time.npy", mmap_mode="r"), dtype=np.int32)
    train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    positions = capped_positions(train_groups)
    offsets = np.concatenate([[0], np.cumsum(train_groups)])
    time_to_group = {int(t): i for i, t in enumerate(train_times[offsets[:-1]])}

    def iter_sections():
        for fold_id, (start, stop) in enumerate((FOLD2, FOLD3)):
            for time_index in range(start, stop):
                rows = positions[time_to_group[time_index]]
                block = np.asarray(train_x[rows], dtype=np.float32)
                y_rank = rank01(np.asarray(train_y[rows], dtype=np.float32))
                yield fold_id, time_index, block, y_rank

    print("[screen] pass 1: base feature ICs", flush=True)
    sums = np.zeros((2, n_base), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for fold_id, _, block, y_rank in iter_sections():
        ranks = rank_columns(block[:, base_positions]).astype(np.float64)
        sums[fold_id] += col_corr(ranks, y_rank.astype(np.float64))
        counts[fold_id] += 1
    t1_f2, t1_f3 = sums[0] / counts[0], sums[1] / counts[1]
    t1_mean = 0.5 * (t1_f2 + t1_f3)
    top = list(np.argsort(-np.abs(t1_mean))[:TOP_T])
    print(f"[screen] top{TOP_T} base columns by |pooled IC|: {top}", flush=True)

    pairs = [(int(a), int(b)) for i, a in enumerate(top) for b in top[i + 1:]]
    specs: list[dict[str, object]] = []
    for j in range(n_base):
        specs.append({"kind": "rank", "col": j})
    for a, b in pairs:
        specs.append({"kind": "diff", "a": a, "b": b})
    for a, b in pairs:
        specs.append({"kind": "prod", "a": a, "b": b})
    for j in top:
        for cat in range(9):
            specs.append({"kind": "concept", "col": int(j), "cat": cat})
    for j in top:
        specs.append({"kind": "absdev", "col": int(j)})
    print(f"[screen] pass 2: {len(specs)} candidates", flush=True)

    sums = np.zeros((2, len(specs)), dtype=np.float64)
    counts = np.zeros(2, dtype=np.int64)
    for fold_id, _, block, y_rank in iter_sections():
        ranks = rank_columns(block[:, base_positions]).astype(np.float64)
        cats = block[:, 408:417].astype(np.float64)
        columns = []
        for spec in specs:
            if spec["kind"] == "rank":
                columns.append(ranks[:, spec["col"]])
            elif spec["kind"] == "diff":
                columns.append(ranks[:, spec["a"]] - ranks[:, spec["b"]])
            elif spec["kind"] == "prod":
                columns.append((ranks[:, spec["a"]] - 0.5) * (ranks[:, spec["b"]] - 0.5))
            elif spec["kind"] == "concept":
                columns.append(group_demean(ranks[:, spec["col"]], cats[:, spec["cat"]]))
            elif spec["kind"] == "absdev":
                columns.append(np.abs(ranks[:, spec["col"]] - 0.5))
        matrix = np.stack(columns, axis=1)
        sums[fold_id] += col_corr(matrix, y_rank.astype(np.float64))
        counts[fold_id] += 1
    f2, f3 = sums[0] / counts[0], sums[1] / counts[1]
    pooled = 0.5 * (f2 + f3)

    rows = []
    for index, spec in enumerate(specs):
        same_sign = np.sign(f2[index]) == np.sign(f3[index]) == np.sign(pooled[index])
        rows.append({
            "index": index,
            "name": factor_name(spec),
            "spec": json.dumps(spec, sort_keys=True),
            "fold2_ic": float(f2[index]),
            "fold3_ic": float(f3[index]),
            "pooled_ic": float(pooled[index]),
            "floor_pass": bool(same_sign and abs(pooled[index]) >= IC_FLOOR),
        })
    return rows, base_positions, np.asarray(top)


def panel_columns(candidates: list[dict[str, object]], base_positions: list[int]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    valid_groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    for split, groups in (("train", train_groups), ("valid", valid_groups)):
        x_all = np.load(TREE / f"{split}_X.npy", mmap_mode="r")
        y_all = np.load(COMMON / f"{split}_y.npy", mmap_mode="r")
        offsets = np.concatenate([[0], np.cumsum(groups)])
        for row in candidates:
            index = int(row["index"])
            spec = json.loads(row["spec"])
            key = (index, split)
            if key in out:
                continue
            states, fps = [], []
            for group_index in range(groups.size):
                left, right = int(offsets[group_index]), int(offsets[group_index + 1])
                block = np.asarray(x_all[left:right], dtype=np.float32)
                factor = compute_factor(block, spec, base_positions).astype(np.float64).reshape(-1, 1)
                y = np.asarray(y_all[left:right], dtype=np.float32)
                states.append(section_state(factor))
                fps.append(section_fingerprint(factor, y)[0])
            if split == "train":
                out.setdefault(index, {})["train_state"] = np.stack(states)
                out.setdefault(index, {})["train_fp"] = np.asarray(fps, dtype=np.float32)
            else:
                out.setdefault(index, {})["valid_state"] = np.stack(states)
                out.setdefault(index, {})["valid_fp"] = np.asarray(fps, dtype=np.float32)
            print(f"[panel] {row['name']} {split} done", flush=True)
    return out


class Evaluator:
    def __init__(self, base_positions: list[int], base_panel: dict, cand_columns: dict[int, dict],
                 cand_specs: dict[int, dict], proxies: dict[int, np.ndarray], valid_base,
                 positions, groups_by_split: dict[str, np.ndarray]):
        self.base_positions = base_positions
        self.base_panel = base_panel
        self.cand_columns = cand_columns
        self.cand_specs = cand_specs
        self.proxies = proxies
        self.valid_base = valid_base
        self.positions = positions
        self.groups_by_split = groups_by_split
        self.train_times = base_panel["train"]["times"]
        self.valid_times = base_panel["valid"]["times"]
        self.train_offsets = np.concatenate([[0], np.cumsum(groups_by_split["train"])])
        self.valid_offsets = np.concatenate([[0], np.cumsum(groups_by_split["valid"])])
        self.train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
        self.valid_x = np.load(TREE / "valid_X.npy", mmap_mode="r")
        self.train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
        self.valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")

    def _states(self, split: str, subset: tuple[int, ...]) -> np.ndarray:
        panel = self.base_panel[split]
        blocks = [panel["states"].reshape(panel["states"].shape[0], 6, -1)]
        blocks.extend(self.cand_columns[i][f"{split}_state"][:, :, None] for i in subset)
        stacked = np.concatenate(blocks, axis=2)
        return stacked.reshape(stacked.shape[0], -1)

    def _fps(self, split: str, subset: tuple[int, ...], mask: np.ndarray) -> np.ndarray:
        columns = [self.base_panel[split]["fingerprints"][mask]]
        columns.extend(self.cand_columns[i][f"{split}_fp"][mask][:, None] for i in subset)
        return np.concatenate(columns, axis=1)

    def evaluate(self, subset: tuple[int, ...]) -> dict[str, list[dict[str, float]]]:
        results: dict[str, list[dict[str, float]]] = {}
        specs = [self.cand_specs[i] for i in subset]
        for window_name, split, start, stop, _ in EVAL_WINDOWS:
            panel_times = self.train_times if split == "train" else self.valid_times
            library_mask = self.train_times < start
            lib_states = self._states("train", subset)[library_mask]
            lib_fp = self._fps("train", subset, library_mask)
            query_indices = np.flatnonzero((panel_times >= start) & (panel_times < stop))
            query_states = self._states(split, subset)[query_indices]
            lib_coord, query_coord = retrieval_coordinates(lib_states, query_states, PCA_COMPONENTS)
            rows = []
            for local_index, group_index in enumerate(query_indices):
                time_index = int(panel_times[group_index])
                distances = np.sqrt(np.sum((lib_coord - query_coord[local_index]) ** 2, axis=1))
                nearest = np.argpartition(distances, K - 1)[:K]
                fingerprint = np.mean(lib_fp[nearest], axis=0)
                if split == "train":
                    block = np.asarray(self.train_x[self.positions[group_index]], dtype=np.float32)
                    y = np.asarray(self.train_y[self.positions[group_index]], dtype=np.float32)
                    base = self.proxies[time_index]
                else:
                    left, right = int(self.valid_offsets[group_index]), int(self.valid_offsets[group_index + 1])
                    block = np.asarray(self.valid_x[left:right], dtype=np.float32)
                    y = np.asarray(self.valid_y[left:right], dtype=np.float32)
                    base = np.asarray(self.valid_base[left:right], dtype=np.float32)
                extended = extended_features(block, self.base_positions, specs)
                base_rank = rank01(base)
                base_ic = safe_corr(base_rank, rank01(y))
                correction = correction_score(extended, fingerprint)
                candidate = (1.0 - ALPHA) * base_rank + ALPHA * rank01(correction)
                candidate_ic = safe_corr(rank01(candidate), rank01(y))
                rows.append({"time": time_index, "baseline_ic": base_ic,
                             "candidate_ic": candidate_ic, "delta": candidate_ic - base_ic})
            results[window_name] = rows
        return results

    def random_control_valid(self, subset: tuple[int, ...]) -> float:
        rng = np.random.default_rng(CONTROL_SEED)
        specs = [self.cand_specs[i] for i in subset]
        library_mask = self.train_times < 2918
        lib_fp = self._fps("train", subset, library_mask)
        query_indices = np.flatnonzero(self.valid_times < 3161)
        ics = []
        for group_index in query_indices:
            left, right = int(self.valid_offsets[group_index]), int(self.valid_offsets[group_index + 1])
            block = np.asarray(self.valid_x[left:right], dtype=np.float32)
            y = np.asarray(self.valid_y[left:right], dtype=np.float32)
            base = np.asarray(self.valid_base[left:right], dtype=np.float32)
            extended = extended_features(block, self.base_positions, specs)
            picked = rng.choice(int(library_mask.sum()), size=K, replace=False)
            fingerprint = np.mean(lib_fp[picked], axis=0)
            base_rank = rank01(base)
            correction = correction_score(extended, fingerprint)
            candidate = (1.0 - ALPHA) * base_rank + ALPHA * rank01(correction)
            ics.append(safe_corr(rank01(candidate), rank01(y)))
        return float(np.mean(ics))


def worst_block(deltas: np.ndarray, block: int = 32) -> float:
    step = max(1, block)
    means = [float(np.mean(deltas[s:s + step])) for s in range(0, deltas.size, step)
             if deltas[s:s + step].size]
    return float(np.min(means)) if means else float("nan")


def main() -> int:
    started = time.time()
    hashes_before = {name: sha256(path) for name, path in PROTECTED.items()}
    RESULT.mkdir(parents=True, exist_ok=True)

    screen_rows, base_positions, top_columns = screening()
    with (RESULT / "screening_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(screen_rows[0]))
        writer.writeheader()
        writer.writerows(screen_rows)
    floor_passers = [row for row in screen_rows if row["floor_pass"]]
    floor_passers.sort(key=lambda row: -abs(row["pooled_ic"]))
    greedy_list = floor_passers[:GREEDY_TRY]
    print(f"[screen] floor passers={len(floor_passers)} greedy list={len(greedy_list)}", flush=True)
    with (RESULT / "top_columns.json").open("w", encoding="utf-8") as handle:
        json.dump({"top16_base_columns": [int(c) for c in top_columns]}, handle, indent=2)

    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    valid_groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    proxies = oof_proxy_parts(train_groups)
    valid_base = np.load(VALID_BASE, mmap_mode="r")
    base_panel = {"train": build_panel("train", base_positions),
                  "valid": build_panel("valid", base_positions)}
    cand_specs = {int(row["index"]): json.loads(row["spec"]) for row in greedy_list}
    cand_columns = panel_columns(greedy_list, base_positions) if greedy_list else {}
    evaluator = Evaluator(base_positions, base_panel, cand_columns, cand_specs, proxies,
                          valid_base, capped_positions(train_groups),
                          {"train": train_groups, "valid": valid_groups})

    reference = evaluator.evaluate(())
    ref_f2 = float(np.mean([r["delta"] for r in reference["fold_2"]]))
    ref_f3 = float(np.mean([r["delta"] for r in reference["fold_3"]]))
    ref_pooled = 0.5 * (ref_f2 + ref_f3)
    ref_valid_ic = float(np.mean([r["candidate_ic"] for r in reference["official_valid"]]))
    exp024a = json.loads(EXP024A_METRICS.read_text(encoding="utf-8"))
    lookup = {(row["window"], row["method"]): row for row in exp024a["window_summaries"]}
    replication = {
        "fold_2_delta": (ref_f2, lookup[("fold_2", "retrieval")]["mean_delta"]),
        "fold_3_delta": (ref_f3, lookup[("fold_3", "retrieval")]["mean_delta"]),
        "valid_candidate_ic": (ref_valid_ic, lookup[("official_valid", "retrieval")]["candidate_mean_ic"]),
    }
    replication_match = all(abs(a - b) < 1e-6 for a, b in replication.values())
    print(f"[reference] pooled={ref_pooled:.6f} replication_match={replication_match}", flush=True)

    trace = [{"step": "reference", "added": None, "pooled": ref_pooled, "kept": False}]
    subset: list[int] = []
    best_pooled = ref_pooled
    fails = 0
    for row in greedy_list:
        index = int(row["index"])
        trial = tuple(subset + [index])
        outcome = evaluator.evaluate(trial)
        pooled = 0.5 * (float(np.mean([r["delta"] for r in outcome["fold_2"]]))
                        + float(np.mean([r["delta"] for r in outcome["fold_3"]])))
        keep = pooled > best_pooled + 1e-6
        trace.append({"step": row["name"], "added": index, "pooled": pooled, "kept": keep})
        print(f"[greedy] {row['name']} pooled={pooled:.6f} keep={keep}", flush=True)
        if keep:
            subset = list(trial)
            best_pooled = pooled
        else:
            fails += 1
            if fails >= GREEDY_STOP_AFTER_FAILS or len(subset) >= GREEDY_KEEP_MAX:
                break
        if len(subset) >= GREEDY_KEEP_MAX:
            break

    final = evaluator.evaluate(tuple(subset))
    final_valid_ic = float(np.mean([r["candidate_ic"] for r in final["official_valid"]]))
    ref_valid_rows = sorted(reference["official_valid"], key=lambda r: r["time"])
    final_valid_rows = sorted(final["official_valid"], key=lambda r: r["time"])
    relative = np.asarray([a["candidate_ic"] - b["candidate_ic"]
                           for a, b in zip(final_valid_rows, ref_valid_rows)], dtype=np.float64)
    loo_ratios = []
    for piece in np.array_split(np.arange(relative.size), min(LOO_BLOCKS, relative.size)):
        keep_mask = np.ones(relative.size, dtype=bool)
        keep_mask[piece] = False
        loo_ratios.append(float(np.mean(relative[keep_mask]) > 0))
    loo_positive_ratio = float(np.mean(loo_ratios))
    random_ic = evaluator.random_control_valid(tuple(subset))

    hashes_after = {name: sha256(path) for name, path in PROTECTED.items()}
    pass_checks = {
        "g1_screening_floor_met": bool(len(floor_passers) >= 1),
        "g2_greedy_improvement": bool(best_pooled > ref_pooled + 1e-6),
        "g3_valid_holdout_nonnegative": bool(final_valid_ic >= ref_valid_ic),
        "g4_valid_worst32_relative": bool(worst_block(relative) >= WORST_BLOCK_GATE),
        "g5_valid_loo_positive_ratio": bool(loo_positive_ratio >= LOO_RATIO_GATE),
        "g6_container_sanity_retrieval_above_random": bool(final_valid_ic > random_ic),
        "g7_protected_hashes_unchanged": bool(hashes_before == hashes_after),
    }
    passed = bool(all(pass_checks.values()))

    final_specs = [cand_specs[i] for i in subset]
    (RESULT / "protocol.json").write_text(json.dumps({
        "experiment": "exp_031a_factor_mining_diagnostic",
        "frozen_container_params": {"pca_components": PCA_COMPONENTS, "neighbors": K,
                                    "alpha": ALPHA, "pca_seed": PCA_SEED},
        "final_factor_specs": final_specs,
        "final_factor_names": [factor_name(s) for s in final_specs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "experiment": "exp_031a_factor_mining_diagnostic",
        "decision": "go_to_exp031b" if passed else "stop_factor_mining_route",
        "passed": passed,
        "reference": {"pooled_fold2_fold3": ref_pooled, "fold2": ref_f2, "fold3": ref_f3,
                      "valid_candidate_ic": ref_valid_ic},
        "reference_replication_vs_exp024a": {k: {"exp031": a, "exp024a": b}
                                             for k, (a, b) in replication.items()},
        "reference_replication_match": replication_match,
        "final": {"pooled_fold2_fold3": best_pooled,
                  "valid_candidate_ic": final_valid_ic,
                  "factors": [factor_name(s) for s in final_specs]},
        "valid_relative_delta_mean": float(np.mean(relative)),
        "valid_relative_worst32": worst_block(relative),
        "valid_relative_loo_positive_ratio": loo_positive_ratio,
        "valid_random_control_ic": random_ic,
        "screening": {"candidates": len(screen_rows), "floor_passers": len(floor_passers),
                      "floor": IC_FLOOR, "greedy_tried": len(trace) - 1},
        "greedy_trace": trace,
        "pass_checks": pass_checks,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "test_labels_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if passed else "completed_rejected",
        "causal_status": "compliant_diagnostic_no_test_labels",
        "baseline_note": "reference replicates exp024a retrieval protocol exactly (same seed/params)",
        "formal_submission_overwritten": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with (RESULT / "per_time_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = []
        for window in ("fold_2", "fold_3", "official_valid"):
            for r in final[window]:
                rows.append({"window": window, "time": r["time"], "variant": "final",
                             "baseline_ic": r["baseline_ic"], "candidate_ic": r["candidate_ic"],
                             "delta": r["delta"]})
            for r in reference[window]:
                rows.append({"window": window, "time": r["time"], "variant": "reference",
                             "baseline_ic": r["baseline_ic"], "candidate_ic": r["candidate_ic"],
                             "delta": r["delta"]})
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({k: metrics[k] for k in ("decision", "passed", "reference", "final",
                                              "pass_checks")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
