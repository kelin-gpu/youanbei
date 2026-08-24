from __future__ import annotations

"""exp032 step 2: zero-training residual-IC diagnostic for the raw feature family.

Candidate families (all causal, <=t information only):
  F1 rank01 of the 59 raw features unused by the main line
  F2 per-stock z-score of all 99 features vs trailing 20-section history
  F3 per-stock z-score of all 99 features vs trailing 60-section history
  F4 rank01 of the 40 used raw features (calibration control, not promotable)

For every candidate column and window (fold2/fold3/official valid) we compute
per-section Spearman ICs against y (rawIC) and against the exp021 base residual
rank(y)-rank(base) (residualIC). Gates are preregistered in the decision log.
No training, no prediction output, no protected-file writes.
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
sys.path.insert(0, str(ROOT / "02_experiments"))
sys.path.insert(0, str(ROOT / "02_experiments" / "exp_024_state_retrieved_rank_residual"))

from run_exp024a import (  # noqa: E402
    COMMON, VALID_BASE, sha256, rank01, capped_positions, oof_proxy_parts,
)

RAW = ROOT / "03_cache" / "exp_032_raw_feature_bank" / "raw_num.npy"
RESULT = ROOT / "04_results" / "exp_032a_raw_feature_diagnostic"
SLICE_START, SLICE_STOP = 1885, 3161

SELECTED40 = [8, 11, 57, 41, 90, 68, 39, 40, 73, 47, 53, 72, 48, 74, 86, 38, 50, 3, 42, 71,
              87, 49, 4, 55, 85, 75, 56, 76, 51, 67, 88, 61, 79, 1, 91, 84, 15, 60, 64, 43]
UNUSED59 = [i for i in range(99) if i not in set(SELECTED40)]

FOLD2, FOLD3, VALID = (1945, 2432), (2432, 2918), (2918, 3161)
RAW_IC_FLOOR = 0.01
RESID_IC_FLOOR = 0.005
POOL_MIN = 20
QUALIFIED_MIN = 10
WORST_BLOCK_GATE = -0.002
WORST_BLOCK = 32
Z_WINDOWS = (20, 60)

PROTECTED = {
    "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
    "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
    "exp024b_prediction": ROOT / "04_results" / "exp_024b_retrieval_exploratory" / "prediction_1.npy",
    "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
}


def rank01_nan(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(values)
    if int(finite.sum()) > 1:
        out[finite] = rank01(values[finite])
    return out


def corr_finite(left: np.ndarray, right: np.ndarray) -> float:
    ok = np.isfinite(left) & np.isfinite(right)
    if int(ok.sum()) < 30:
        return np.nan
    a = left[ok].astype(np.float64)
    b = right[ok].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denom) if denom > 1e-12 else np.nan


def zscore_features(raw: np.ndarray, t_local: int, window: int, feature_ids) -> np.ndarray:
    start = max(0, t_local - window)
    history = raw[start:t_local][:, :, feature_ids]
    current = raw[t_local][:, feature_ids]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(history, axis=0)
        std = np.nanstd(history, axis=0)
        std_safe = np.where(std > 1e-9, std, np.nan)
        z = (current - mean) / std_safe
    return z.astype(np.float32)


def section_candidates(raw: np.ndarray, t_local: int, z_cache: dict[int, dict[int, np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for j in UNUSED59:
        out[f"F1_rank_num{j}"] = rank01_nan(raw[t_local][:, j].astype(np.float32))
    for w in Z_WINDOWS:
        z = z_cache[t_local][w]
        for k, j in enumerate(range(99)):
            out[f"F{2 if w == 20 else 3}_z{w}_num{j}"] = z[:, k]
    for j in SELECTED40:
        out[f"F4_rank_num{j}"] = rank01_nan(raw[t_local][:, j].astype(np.float32))
    return out


def main() -> int:
    started = time.time()
    hashes_before = {name: sha256(path) for name, path in PROTECTED.items()}
    RESULT.mkdir(parents=True, exist_ok=True)
    raw = np.load(RAW, mmap_mode="r")

    train_groups = np.asarray(np.load(COMMON / "train_group_sizes.npy"), dtype=np.int64)
    valid_groups = np.asarray(np.load(COMMON / "valid_group_sizes.npy"), dtype=np.int64)
    train_times = np.asarray(np.load(COMMON / "train_time.npy", mmap_mode="r"), dtype=np.int32)
    train_stocks = np.asarray(np.load(COMMON / "train_stock.npy", mmap_mode="r"), dtype=np.int32)
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    valid_times = np.asarray(np.load(COMMON / "valid_time.npy", mmap_mode="r"), dtype=np.int32)
    valid_stocks = np.asarray(np.load(COMMON / "valid_stock.npy", mmap_mode="r"), dtype=np.int32)
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")

    print("[diag] building OOF proxies for fold windows", flush=True)
    proxies = oof_proxy_parts(train_groups)
    valid_base = np.asarray(np.load(VALID_BASE, mmap_mode="r"), dtype=np.float32)
    train_offsets = np.concatenate([[0], np.cumsum(train_groups)])
    valid_offsets = np.concatenate([[0], np.cumsum(valid_groups)])
    time_to_group = {int(t): i for i, t in enumerate(train_times[train_offsets[:-1]])}
    positions = capped_positions(train_groups)

    all_features = ([f"F1_rank_num{j}" for j in UNUSED59]
                    + [f"F2_z20_num{j}" for j in range(99)]
                    + [f"F3_z60_num{j}" for j in range(99)]
                    + [f"F4_rank_num{j}" for j in SELECTED40])
    feature_index = {name: i for i, name in enumerate(all_features)}
    windows = {"fold2": (FOLD2, "train"), "fold3": (FOLD3, "train"), "valid": (VALID, "valid")}

    raw_sums = {w: np.zeros(len(all_features)) for w in windows}
    resid_sums = {w: np.zeros(len(all_features)) for w in windows}
    counts = {w: 0 for w in windows}
    valid_section_residual_ics: dict[str, list[float]] = {}

    z_cache: dict[int, dict[int, np.ndarray]] = {}

    def ensure_z(t_local: int) -> None:
        if t_local in z_cache:
            return
        z_cache[t_local] = {w: zscore_features(raw, t_local, w, list(range(99))) for w in Z_WINDOWS}
        stale = [k for k in z_cache if k < t_local - max(Z_WINDOWS)]
        for k in stale:
            del z_cache[k]

    for window_name, ((start, stop), split) in windows.items():
        print(f"[diag] window {window_name} [{start},{stop})", flush=True)
        for time_index in range(start, stop):
            t_local = time_index - SLICE_START
            ensure_z(t_local)
            cands = section_candidates(raw, t_local, z_cache)
            if split == "train":
                group_index = time_to_group[time_index]
                rows = positions[group_index]
                stocks = train_stocks[rows]
                y = np.asarray(train_y[rows], dtype=np.float32)
                base = np.asarray(proxies[time_index], dtype=np.float32)
            else:
                group_index = int(np.flatnonzero(valid_times[valid_offsets[:-1]] == time_index)[0])
                left, right = int(valid_offsets[group_index]), int(valid_offsets[group_index + 1])
                stocks = valid_stocks[left:right]
                y = np.asarray(valid_y[left:right], dtype=np.float32)
                base = valid_base[left:right]
            y_rank = rank01(y)
            resid = y_rank - rank01(base)
            for name, values in cands.items():
                col = values[stocks]
                if np.all(~np.isfinite(col)):
                    continue
                idx = feature_index[name]
                ric = corr_finite(col, y_rank)
                eic = corr_finite(col, resid)
                if np.isfinite(ric):
                    raw_sums[window_name][idx] += ric
                if np.isfinite(eic):
                    resid_sums[window_name][idx] += eic
                if window_name == "valid" and np.isfinite(eic):
                    valid_section_residual_ics.setdefault(name, []).append(eic)
            counts[window_name] += 1
            if counts[window_name] % 100 == 0:
                print(f"  {counts[window_name]} sections done", flush=True)

    for w in windows:
        raw_sums[w] /= counts[w]
        resid_sums[w] /= counts[w]

    rows = []
    for name in all_features:
        idx = feature_index[name]
        pooled_raw = 0.5 * (raw_sums["fold2"][idx] + raw_sums["fold3"][idx])
        sign_consistent = (np.sign(raw_sums["fold2"][idx]) == np.sign(raw_sums["fold3"][idx])
                           == np.sign(pooled_raw) != 0)
        rows.append({
            "feature": name,
            "rawIC_fold2": float(raw_sums["fold2"][idx]),
            "rawIC_fold3": float(raw_sums["fold3"][idx]),
            "rawIC_pooled": float(pooled_raw),
            "rawIC_valid": float(raw_sums["valid"][idx]),
            "residIC_fold2": float(resid_sums["fold2"][idx]),
            "residIC_fold3": float(resid_sums["fold3"][idx]),
            "residIC_valid": float(resid_sums["valid"][idx]),
            "pool_pass": bool(sign_consistent and abs(pooled_raw) >= RAW_IC_FLOOR),
        })
    with (RESULT / "feature_ic_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pool = [r for r in rows if r["pool_pass"] and not r["feature"].startswith("F4")]
    qualified = []
    for r in pool:
        direction = np.sign(r["rawIC_pooled"])
        same_direction = all(np.sign(r[f"residIC_{w}"]) == direction
                             for w in ("fold2", "fold3", "valid"))
        strong = all(abs(r[f"residIC_{w}"]) >= RESID_IC_FLOOR
                     for w in ("fold2", "fold3", "valid"))
        if same_direction and strong:
            qualified.append(r)

    valid_mean = (float(np.mean([r["residIC_valid"] for r in qualified]))
                  if qualified else float("nan"))
    worst_block = float("nan")
    loo_ratio = float("nan")
    if qualified:
        family_series = {}
        for r in qualified:
            for k, v in enumerate(valid_section_residual_ics.get(r["feature"], [])):
                family_series[k] = family_series.get(k, 0.0) + v
        series = np.asarray([family_series[k] / len(qualified)
                             for k in sorted(family_series)])
        blocks = [float(np.mean(series[s:s + WORST_BLOCK]))
                  for s in range(0, series.size, WORST_BLOCK) if series[s:s + WORST_BLOCK].size]
        worst_block = min(blocks)
        pieces = np.array_split(np.arange(series.size), 8)
        loo_ratio = float(np.mean([float(np.mean(np.delete(series, p)) > 0) for p in pieces]))

    hashes_after = {name: sha256(path) for name, path in PROTECTED.items()}
    gates = {
        "g1_signal_pool_exists": bool(len(pool) >= POOL_MIN),
        "g2_residual_signal": bool(len(qualified) >= QUALIFIED_MIN),
        "g3_valid_transfer": bool(np.isfinite(valid_mean) and valid_mean > 0),
        "g4_stability_worst_block": bool(np.isfinite(worst_block) and worst_block >= WORST_BLOCK_GATE),
        "g5_protected_hashes_unchanged": bool(hashes_before == hashes_after),
    }
    passed = all(gates.values())

    metrics = {
        "experiment": "exp_032a_raw_feature_diagnostic",
        "decision": "go_phase_d_extended_tabular" if passed else (
            "raw_family_route_closed_no_signal" if not gates["g1_signal_pool_exists"]
            else "no_orthogonal_signal" if not gates["g2_residual_signal"]
            else "local_only_not_transferable" if not gates["g3_valid_transfer"]
            else "unstable_signal"),
        "passed": passed,
        "gates": gates,
        "counts": {"candidates_total": len(rows), "pool_pass": len(pool), "qualified": len(qualified)},
        "family_breakdown": {
            "F1_rank_unused59": len([r for r in pool if r["feature"].startswith("F1")]),
            "F2_z20": len([r for r in pool if r["feature"].startswith("F2")]),
            "F3_z60": len([r for r in pool if r["feature"].startswith("F3")]),
        },
        "qualified_features": [
            {"feature": r["feature"], "rawIC_pooled": round(r["rawIC_pooled"], 5),
             "residIC_valid": round(r["residIC_valid"], 5)} for r in qualified],
        "valid_residual_mean": valid_mean,
        "valid_worst32_block": worst_block,
        "valid_loo_positive_ratio": loo_ratio,
        "sections_per_window": counts,
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
        "formal_submission_overwritten": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in
                      ("decision", "gates", "counts", "family_breakdown",
                       "valid_residual_mean", "valid_worst32_block",
                       "valid_loo_positive_ratio")}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
