from __future__ import annotations

"""exp029a: adversarial validation + four-window feature-exposure neutralization diagnostic.

Part A ranks the 408 numeric tree features by train-vs-test adversarial importance (labels never
used). Part B evaluates per-section linear neutralization of the baseline rank against the
cross-sectional ranks of the top-N drifted features, on the frozen exp024a evaluation windows.
Parameters are selected on fold2+fold3 only; official valid is a pure holdout. The script never
loads Test labels, never writes a prediction, and never touches protected files.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "02_experiments"))
sys.path.insert(0, str(ROOT / "02_experiments" / "exp_024_state_retrieved_rank_residual"))

from run_exp024a import (  # noqa: E402
    COMMON, TREE, VALID_BASE, EVAL_WINDOWS,
    sha256, safe_corr, rank01, capped_positions, oof_proxy_parts,
)

RESULT = ROOT / "04_results" / "exp_029a_neutralization_diagnostic"
REGISTRY = ROOT / "04_results" / "_acceptance" / "drift_feature_registry.csv"
MANIFEST = ROOT / "03_cache" / "processed_data_v1" / "manifest.json"
PROTECTED = {
    "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
    "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
    "exp024b_prediction": ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy",
    "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
}

LAMBDA_GRID = (0.05, 0.10, 0.15)
N_GRID = (10, 20, 40)
MAX_N = max(N_GRID)
SEED = 20260824
MAX_ROWS_PER_SIDE = 200_000
AUC_GATE = 0.55
RANK_CORR_GATE = 0.995
WORST_BLOCK_GATE = -0.002
LOO_BLOCKS = 8
LOO_POSITIVE_RATIO_GATE = 0.75
BLOCK_SIZE = 32


def numeric_feature_names() -> list[str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = list(manifest["features"]["numeric_names"])
    if len(names) != 408:
        raise RuntimeError(f"expected 408 numeric features, got {len(names)}")
    return names


def registry_actions() -> dict[str, str]:
    actions: dict[str, str] = {}
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            actions[row["feature"]] = row["audit_action"]
    return actions


def stratified_rows(groups: np.ndarray, per_side_cap: int) -> np.ndarray:
    quota = max(1, per_side_cap // int(groups.size))
    picks, offset = [], 0
    for size_value in groups:
        size = int(size_value)
        take = min(size, quota)
        picks.append(offset + np.linspace(0, size - 1, take, dtype=np.int64))
        offset += size
    return np.concatenate(picks)


def adversarial_validation(feature_names: list[str]) -> tuple[float, list[dict[str, object]]]:
    import lightgbm as lgb

    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    test_groups = np.asarray(np.load(COMMON / "test_group_sizes.npy"), dtype=np.int64)
    train_rows = stratified_rows(train_groups, MAX_ROWS_PER_SIDE)
    test_rows = stratified_rows(test_groups, MAX_ROWS_PER_SIDE)
    print(f"[adversarial] sampled train={train_rows.size} test={test_rows.size} rows", flush=True)

    train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
    test_x = np.load(TREE / "test_X.npy", mmap_mode="r")
    numeric = slice(0, 408)
    left = np.nan_to_num(np.asarray(train_x[train_rows][:, numeric], dtype=np.float32))
    right = np.nan_to_num(np.asarray(test_x[test_rows][:, numeric], dtype=np.float32))
    x = np.concatenate([left, right])
    y = np.concatenate([np.zeros(left.shape[0], dtype=np.int8), np.ones(right.shape[0], dtype=np.int8)])
    print(f"[adversarial] matrix {x.shape}", flush=True)

    importance = np.zeros(x.shape[1], dtype=np.float64)
    aucs: list[float] = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (fit_part, val_part) in enumerate(splitter.split(x, y)):
        model = lgb.LGBMClassifier(
            n_estimators=250, learning_rate=0.05, num_leaves=63, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        model.fit(x[fit_part], y[fit_part])
        proba = model.predict_proba(x[val_part])[:, 1]
        aucs.append(float(roc_auc_score(y[val_part], proba)))
        importance += np.asarray(model.booster_.feature_importance("gain"), dtype=np.float64)
        print(f"[adversarial] fold {fold + 1}/5 auc={aucs[-1]:.4f}", flush=True)
    importance /= len(aucs)
    auc = float(np.mean(aucs))

    actions = registry_actions()
    order = np.argsort(-importance)
    rows: list[dict[str, object]] = []
    for rank, column in enumerate(order):
        name = feature_names[int(column)]
        rows.append({
            "rank": rank + 1,
            "column": int(column),
            "feature": name,
            "gain_importance": float(importance[int(column)]),
            "registry_action": actions.get(name, "missing"),
        })
    return auc, rows


def column_rank01(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], 0.5, dtype=np.float64)
    finite = np.isfinite(values)
    if int(finite.sum()) >= 2:
        out[finite] = rankdata(values[finite], method="average") - 1.0
        out[finite] /= max(1, int(finite.sum()) - 1)
    return out


def section_feature_ranks(x: np.ndarray, columns: np.ndarray) -> np.ndarray:
    ranks = np.empty((x.shape[0], columns.size), dtype=np.float64)
    for position, column in enumerate(columns):
        ranks[:, position] = column_rank01(np.asarray(x[:, column], dtype=np.float64))
    ranks -= ranks.mean(axis=0, keepdims=True)
    return ranks


def worst_block_mean(deltas: np.ndarray, block: int = BLOCK_SIZE) -> float:
    if deltas.size == 0:
        return float("nan")
    step = max(1, block)
    means = [float(np.mean(deltas[start:start + step])) for start in range(0, deltas.size, step)
             if deltas[start:start + step].size > 0]
    return float(np.min(means))


def main() -> int:
    started = time.time()
    hashes_before = {name: sha256(path) for name, path in PROTECTED.items()}
    RESULT.mkdir(parents=True, exist_ok=True)

    feature_names = numeric_feature_names()
    auc, adversarial_rows = adversarial_validation(feature_names)
    with (RESULT / "adversarial_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adversarial_rows[0]))
        writer.writeheader()
        writer.writerows(adversarial_rows)

    top_columns = np.asarray([int(row["column"]) for row in adversarial_rows[:MAX_N]], dtype=np.int64)
    top_names = [str(row["feature"]) for row in adversarial_rows[:MAX_N]]
    print(f"[adversarial] auc={auc:.4f} gate={AUC_GATE}", flush=True)

    grouped: dict[str, list[float]] = {}
    for row in adversarial_rows:
        grouped.setdefault(str(row["registry_action"]), []).append(float(row["gain_importance"]))
    importance_by_action = {key: float(np.mean(val)) for key, val in grouped.items()}
    print(f"[adversarial] mean gain by registry action: {importance_by_action}", flush=True)

    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    proxy = oof_proxy_parts(train_groups)
    valid_base = np.load(VALID_BASE, mmap_mode="r")
    train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
    valid_x = np.load(TREE / "valid_X.npy", mmap_mode="r")
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    train_times_all = np.load(COMMON / "train_time.npy", mmap_mode="r")
    valid_times_all = np.load(COMMON / "valid_time.npy", mmap_mode="r")
    train_positions = capped_positions(train_groups)
    train_offsets = np.concatenate([[0], np.cumsum(train_groups)])
    valid_groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    valid_offsets = np.concatenate([[0], np.cumsum(valid_groups)])

    records: list[dict[str, object]] = []
    for window_name, split, start, stop, baseline_name in EVAL_WINDOWS:
        panel_times = train_times_all if split == "train" else valid_times_all
        group_count = train_groups.size if split == "train" else valid_groups.size
        base_index = start - 486 if split == "train" else start - 2918
        for group_index in range(base_index, base_index + (stop - start)):
            time_index = int(panel_times[int(train_offsets[group_index]) if split == "train"
                                          else int(valid_offsets[group_index])])
            if split == "train":
                rows_idx = train_positions[group_index]
                x = np.asarray(train_x[rows_idx[:, None], top_columns[None, :]], dtype=np.float32)
                y = np.asarray(train_y[rows_idx], dtype=np.float32)
                base = proxy[time_index]
            else:
                left, right = int(valid_offsets[group_index]), int(valid_offsets[group_index + 1])
                x = np.asarray(valid_x[left:right][:, top_columns], dtype=np.float32)
                y = np.asarray(valid_y[left:right], dtype=np.float32)
                base = np.asarray(valid_base[left:right], dtype=np.float32)
            if x.shape[0] < 10:
                continue

            base_rank = rank01(base)
            y_rank = rank01(y)
            baseline_ic = safe_corr(base_rank, y_rank)
            feature_ranks = section_feature_ranks(x, np.arange(top_columns.size))
            betas = {n: np.linalg.lstsq(feature_ranks[:, :n], base_rank, rcond=None)[0] for n in N_GRID}
            for n in N_GRID:
                projection = feature_ranks[:, :n] @ betas[n]
                for lam in LAMBDA_GRID:
                    candidate = rank01(base_rank - lam * projection)
                    correlation = safe_corr(base_rank, candidate)
                    records.append({
                        "window": window_name,
                        "time": time_index,
                        "baseline": baseline_name,
                        "lambda": lam,
                        "n_features": n,
                        "baseline_ic": baseline_ic,
                        "candidate_ic": safe_corr(candidate, y_rank),
                        "delta": safe_corr(candidate, y_rank) - baseline_ic,
                        "rank_correlation_to_baseline": correlation,
                    })
        done = sum(1 for r in records if r["window"] == window_name)
        print(f"[{window_name}] evaluated {done} section-combos", flush=True)

    with (RESULT / "per_time_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    def window_stats(window: str, lam: float, n: int) -> dict[str, float]:
        chosen = [r for r in records if r["window"] == window and r["lambda"] == lam and r["n_features"] == n]
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
    for lam in LAMBDA_GRID:
        for n in N_GRID:
            fold2 = window_stats("fold_2", lam, n)
            fold3 = window_stats("fold_3", lam, n)
            constraints = (
                fold2["mean_delta"] > 0 and fold3["mean_delta"] > 0
                and fold2["min_rank_correlation"] >= RANK_CORR_GATE
                and fold3["min_rank_correlation"] >= RANK_CORR_GATE
                and fold2["worst_block_mean"] >= WORST_BLOCK_GATE
                and fold3["worst_block_mean"] >= WORST_BLOCK_GATE
            )
            pooled = 0.5 * (fold2["mean_delta"] + fold3["mean_delta"])
            selection.append({
                "lambda": lam, "n_features": n, "pooled_fold2_fold3_delta": pooled,
                "fold2": fold2, "fold3": fold3, "constraints_passed": bool(constraints),
            })
            if constraints:
                eligible.append((pooled, lam, n))

    if not eligible:
        chosen_lam, chosen_n, chosen_pooled = None, None, None
        valid_stats = None
        pass_checks = {"any_combo_eligible_on_folds": False}
    else:
        eligible.sort(reverse=True)
        chosen_pooled, chosen_lam, chosen_n = eligible[0]
        valid_stats = window_stats("official_valid", chosen_lam, chosen_n)
        chosen_deltas = np.asarray([
            r["delta"] for r in records
            if r["lambda"] == chosen_lam and r["n_features"] == chosen_n
        ], dtype=np.float64)
        loo_means = []
        for piece in np.array_split(np.arange(chosen_deltas.size), min(LOO_BLOCKS, chosen_deltas.size)):
            keep = np.ones(chosen_deltas.size, dtype=bool)
            keep[piece] = False
            loo_means.append(float(np.mean(chosen_deltas[keep])))
        loo_positive_ratio = float(np.mean(np.asarray(loo_means) > 0))

        hashes_after = {name: sha256(path) for name, path in PROTECTED.items()}
        pass_checks = {
            "adversarial_auc_at_least_0_55": bool(auc >= AUC_GATE),
            "any_combo_eligible_on_folds": True,
            "official_valid_pooled_delta_nonnegative": bool(valid_stats["mean_delta"] >= 0.0),
            "official_valid_worst_block_at_least_-0_002": bool(valid_stats["worst_block_mean"] >= WORST_BLOCK_GATE),
            "official_valid_rank_correlation_min_0_995": bool(valid_stats["min_rank_correlation"] >= RANK_CORR_GATE),
            "loo_delta_positive_ratio_at_least_0_75": bool(loo_positive_ratio >= LOO_POSITIVE_RATIO_GATE),
            "protected_hashes_unchanged": bool(hashes_before == hashes_after),
        }

    passed = bool(pass_checks.get("adversarial_auc_at_least_0_55") and all(pass_checks.values()))
    protocol = {
        "experiment": "exp_029a_neutralization_diagnostic",
        "frozen_parameters": {
            "lambda_grid": list(LAMBDA_GRID),
            "n_grid": list(N_GRID),
            "selection_rule": "max pooled fold2+fold3 mean delta subject to fold2>0, fold3>0, min rank corr>=0.995, worst 32-block>=-0.002 on both folds",
            "adversarial_auc": auc,
            "adversarial_seed": SEED,
            "top_features": top_names,
            "chosen_lambda": chosen_lam,
            "chosen_n_features": chosen_n,
            "pooled_fold2_fold3_delta": chosen_pooled,
            "preserve_exp024b_first_sections": 6,
        },
    }
    (RESULT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "experiment": "exp_029a_neutralization_diagnostic",
        "decision": "go_to_exp029b" if passed else "stop_neutralization_route",
        "passed": passed,
        "adversarial_validation": {
            "auc_mean": auc,
            "mean_gain_by_registry_action": importance_by_action,
            "top10_features": top_names[:10],
        },
        "selection_grid": selection,
        "chosen_parameters": {"lambda": chosen_lam, "n_features": chosen_n,
                              "pooled_fold2_fold3_delta": chosen_pooled},
        "official_valid_stats": valid_stats,
        "pass_checks": pass_checks,
        "gates": {
            "auc_gate": AUC_GATE, "rank_correlation_gate": RANK_CORR_GATE,
            "worst_block_gate": WORST_BLOCK_GATE, "loo_positive_ratio_gate": LOO_POSITIVE_RATIO_GATE,
        },
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": {name: sha256(path) for name, path in PROTECTED.items()},
        "protected_unchanged": bool(hashes_before == {name: sha256(path) for name, path in PROTECTED.items()}),
        "test_labels_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if passed else "completed_rejected",
        "causal_status": "compliant_diagnostic_no_test_labels",
        "baseline_note": "fold windows use exp016 static strict-OOF proxy; official valid uses frozen exp021 full prediction; feature axis neutralization differs from rejected entity-axis exp028a",
        "formal_submission_overwritten": False,
        "test_prediction_generated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in
                      ("decision", "passed", "chosen_parameters", "pass_checks")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
