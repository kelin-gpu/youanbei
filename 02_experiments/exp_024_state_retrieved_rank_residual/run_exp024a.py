from __future__ import annotations

"""Strictly causal, prediction-free diagnostic for historical state retrieval.

The script never opens Test arrays and never writes a submission.  It compares
state-similar historical sections with equally-sized recent and random controls.
Official Valid uses the frozen exp021 full-panel prediction.  Earlier windows
use a static-rank blend of the existing exp016 strict-OOF family matrix and are
reported explicitly as a proxy baseline.
"""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_experiments"))

from exp_016_unified_expert_fusion.config import BASE_WEIGHTS, FAMILIES  # noqa: E402


COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
TREE = ROOT / "03_cache" / "processed_data_v1" / "tree"
OOF = ROOT / "03_cache" / "exp_016_unified_expert_fusion" / "oof_predictions" / "family_matrix.npy"
VALID_BASE = ROOT / "04_results" / "_acceptance" / "exp021_validation_audit" / "full_valid_prediction.npy"
RESULT = ROOT / "04_results" / "exp_024a_retrieval_diagnostic"
PROTOCOL = RESULT / "protocol.json"
REGISTRY = ROOT / "04_results" / "_acceptance" / "drift_feature_registry.csv"
MANIFEST = ROOT / "03_cache" / "processed_data_v1" / "manifest.json"

EVAL_WINDOWS = (
    ("fold_1", "train", 1459, 1945, "exp016_static_strict_oof_proxy"),
    ("fold_2", "train", 1945, 2432, "exp016_static_strict_oof_proxy"),
    ("fold_3", "train", 2432, 2918, "exp016_static_strict_oof_proxy"),
    ("official_valid", "valid", 2918, 3161, "exp021_full_valid_frozen"),
)
OOF_START = 1459


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 10:
        return 0.0
    left = left[finite] - np.mean(left[finite])
    right = right[finite] - np.mean(right[finite])
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denom) if denom > 1e-12 else 0.0


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    ranked = rankdata(values, method="average")
    return ((ranked - 1.0) / max(1, ranked.size - 1)).astype(np.float32)


def rank_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    ranked = rankdata(values, axis=0, method="average")
    denom = max(1, values.shape[0] - 1)
    return ((ranked - 1.0) / denom).astype(np.float32)


def feature_contract() -> tuple[list[int], list[str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tree_names = list(manifest["features"]["numeric_names"][:40])
    actions: dict[str, str] = {}
    with REGISTRY.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            actions[row["feature"]] = row["audit_action"]
    allowed = {"stable_keep", "rank_or_robust_transform"}
    positions = [index for index, name in enumerate(tree_names) if actions.get(name) in allowed]
    names = [tree_names[index] for index in positions]
    if len(positions) < 10:
        raise RuntimeError("Too few preregistered stable features")
    return positions, names


def section_state(x: np.ndarray) -> np.ndarray:
    q10, median, q90 = np.nanquantile(x, [0.10, 0.50, 0.90], axis=0)
    q25, q75 = np.nanquantile(x, [0.25, 0.75], axis=0)
    std = np.nanstd(x, axis=0)
    zero_rate = np.mean(np.isclose(x, 0.0), axis=0)
    return np.concatenate([q10, median, q90, q75 - q25, std, zero_rate]).astype(np.float32)


def section_fingerprint(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xr = rank_columns(x).astype(np.float64)
    yr = rank01(y).astype(np.float64)
    xr -= xr.mean(axis=0, keepdims=True)
    yr -= yr.mean()
    denom = np.sqrt(np.sum(xr * xr, axis=0) * np.sum(yr * yr))
    out = np.divide(np.sum(xr * yr[:, None], axis=0), denom,
                    out=np.zeros(xr.shape[1], dtype=np.float64), where=denom > 1e-12)
    return out.astype(np.float32)


def build_panel(split: str, feature_positions: list[int]) -> dict[str, np.ndarray]:
    x_all = np.load(TREE / f"{split}_X.npy", mmap_mode="r")
    y_all = np.load(COMMON / f"{split}_y.npy", mmap_mode="r")
    times_all = np.load(COMMON / f"{split}_time.npy", mmap_mode="r")
    groups = np.asarray(np.load(COMMON / f"{split}_group_sizes.npy"), dtype=np.int64)
    states, fingerprints, times = [], [], []
    offset = 0
    for group_index, size_value in enumerate(groups):
        size = int(size_value)
        x = np.asarray(x_all[offset:offset + size, feature_positions], dtype=np.float32)
        y = np.asarray(y_all[offset:offset + size], dtype=np.float32)
        states.append(section_state(x))
        fingerprints.append(section_fingerprint(x, y))
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


def capped_positions(groups: np.ndarray, cap: int = 1024) -> list[np.ndarray]:
    parts, offset = [], 0
    for size_value in groups:
        size = int(size_value)
        take = min(size, cap)
        parts.append(offset + np.linspace(0, size - 1, take, dtype=np.int64))
        offset += size
    return parts


def oof_proxy_parts(train_groups: np.ndarray) -> dict[int, np.ndarray]:
    matrix = np.load(OOF, mmap_mode="r")
    weights = np.asarray([BASE_WEIGHTS[name] for name in FAMILIES], dtype=np.float64)
    weights /= weights.sum()
    result: dict[int, np.ndarray] = {}
    offset = 0
    for time_index in range(OOF_START, 2918):
        group_pos = time_index - 486
        size = min(int(train_groups[group_pos]), 1024)
        block = np.asarray(matrix[offset:offset + size], dtype=np.float32)
        ranked = rank_columns(block)
        result[time_index] = np.asarray(ranked @ weights, dtype=np.float32)
        offset += size
    if offset != matrix.shape[0]:
        raise RuntimeError("OOF family matrix alignment failure")
    return result


def retrieval_coordinates(library_states: np.ndarray, query_states: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(library_states)
    library_scaled = np.nan_to_num(scaler.transform(library_states), nan=0.0, posinf=0.0, neginf=0.0)
    query_scaled = np.nan_to_num(scaler.transform(query_states), nan=0.0, posinf=0.0, neginf=0.0)
    count = min(int(components), library_scaled.shape[0] - 1, library_scaled.shape[1])
    pca = PCA(n_components=max(1, count), random_state=20260823).fit(library_scaled)
    return pca.transform(library_scaled).astype(np.float32), pca.transform(query_scaled).astype(np.float32)


def correction_score(x: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
    ranked = rank_columns(x).astype(np.float64)
    weights = np.asarray(fingerprint, dtype=np.float64)
    scale = float(np.sum(np.abs(weights)))
    if scale <= 1e-12:
        return np.full(x.shape[0], 0.5, dtype=np.float32)
    return np.asarray(ranked @ (weights / scale), dtype=np.float32)


def loo_confidence_stats(confidence: np.ndarray, delta: np.ndarray, blocks: int) -> dict[str, object]:
    indices = np.arange(delta.size)
    pieces = np.array_split(indices, min(blocks, delta.size))
    correlations = []
    for piece in pieces:
        keep = np.ones(delta.size, dtype=bool)
        keep[piece] = False
        corr = spearmanr(confidence[keep], delta[keep], nan_policy="omit").statistic
        correlations.append(float(corr) if np.isfinite(corr) else 0.0)
    arr = np.asarray(correlations, dtype=np.float64)
    return {
        "correlations": correlations,
        "median": float(np.median(arr)),
        "positive_ratio": float(np.mean(arr > 0)),
    }


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    params = protocol["preregistered_parameters"]
    k = int(params["neighbors"])
    alpha = float(params["diagnostic_blend_alpha"])
    rng = np.random.default_rng(int(params["random_seed"]))
    protected = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    hashes_before = {name: sha256(path) for name, path in protected.items()}

    feature_positions, feature_names = feature_contract()
    print(f"eligible features={len(feature_positions)}", flush=True)
    train = build_panel("train", feature_positions)
    valid = build_panel("valid", feature_positions)
    proxy = oof_proxy_parts(train["groups"])
    valid_base = np.load(VALID_BASE, mmap_mode="r")
    train_x = np.load(TREE / "train_X.npy", mmap_mode="r")
    valid_x = np.load(TREE / "valid_X.npy", mmap_mode="r")
    train_y = np.load(COMMON / "train_y.npy", mmap_mode="r")
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    train_positions = capped_positions(train["groups"])

    train_offsets = np.concatenate([[0], np.cumsum(train["groups"])])
    valid_offsets = np.concatenate([[0], np.cumsum(valid["groups"])])
    rows: list[dict[str, object]] = []

    for window_name, split, start, stop, baseline_name in EVAL_WINDOWS:
        library_mask = train["times"] < start
        library_states = train["states"][library_mask]
        library_fp = train["fingerprints"][library_mask]
        library_times = train["times"][library_mask]
        panel = train if split == "train" else valid
        query_mask = (panel["times"] >= start) & (panel["times"] < stop)
        query_indices = np.flatnonzero(query_mask)
        library_coord, query_coord = retrieval_coordinates(
            library_states, panel["states"][query_mask], int(params["state_pca_components"])
        )
        print(f"[{window_name}] library={library_states.shape[0]} query={query_indices.size}", flush=True)

        for local_index, group_index in enumerate(query_indices):
            time_index = int(panel["times"][group_index])
            distances = np.sqrt(np.sum((library_coord - query_coord[local_index]) ** 2, axis=1))
            nearest = np.argpartition(distances, k - 1)[:k]
            recent = np.arange(max(0, library_fp.shape[0] - k), library_fp.shape[0])
            random = rng.choice(library_fp.shape[0], size=k, replace=False)
            methods = {"retrieval": nearest, "recent": recent, "random": random}

            if split == "train":
                relative = group_index
                absolute = train_positions[relative]
                x = np.asarray(train_x[absolute[:, None], np.asarray(feature_positions)[None, :]], dtype=np.float32)
                y = np.asarray(train_y[absolute], dtype=np.float32)
                base = proxy[time_index]
            else:
                left, right = int(valid_offsets[group_index]), int(valid_offsets[group_index + 1])
                x = np.asarray(valid_x[left:right, feature_positions], dtype=np.float32)
                y = np.asarray(valid_y[left:right], dtype=np.float32)
                base = np.asarray(valid_base[left:right], dtype=np.float32)

            base_rank = rank01(base)
            base_ic = safe_corr(base_rank, rank01(y))
            for method, picked in methods.items():
                fingerprint = np.mean(library_fp[picked], axis=0)
                correction = correction_score(x, fingerprint)
                candidate = (1.0 - alpha) * base_rank + alpha * rank01(correction)
                candidate_ic = safe_corr(rank01(candidate), rank01(y))
                fp_std = float(np.mean(np.std(library_fp[picked], axis=0)))
                mean_distance = float(np.mean(distances[picked])) if method == "retrieval" else float("nan")
                confidence = -(mean_distance + fp_std) if method == "retrieval" else float("nan")
                rows.append({
                    "window": window_name,
                    "split": split,
                    "time": time_index,
                    "baseline": baseline_name,
                    "method": method,
                    "baseline_ic": base_ic,
                    "correction_ic": safe_corr(rank01(correction), rank01(y)),
                    "candidate_ic": candidate_ic,
                    "delta": candidate_ic - base_ic,
                    "confidence": confidence,
                    "mean_neighbor_distance": mean_distance,
                    "fingerprint_std": fp_std,
                    "min_library_time": int(np.min(library_times[picked])),
                    "max_library_time": int(np.max(library_times[picked])),
                    "library_cutoff_exclusive": start,
                })

    summaries: list[dict[str, object]] = []
    for window_name, _, _, _, baseline_name in EVAL_WINDOWS:
        for method in ("retrieval", "recent", "random"):
            selected = [row for row in rows if row["window"] == window_name and row["method"] == method]
            delta = np.asarray([row["delta"] for row in selected], dtype=np.float64)
            summaries.append({
                "window": window_name,
                "baseline": baseline_name,
                "method": method,
                "sections": len(selected),
                "baseline_mean_ic": float(np.mean([row["baseline_ic"] for row in selected])),
                "candidate_mean_ic": float(np.mean([row["candidate_ic"] for row in selected])),
                "mean_delta": float(np.mean(delta)),
                "positive_delta_ratio": float(np.mean(delta > 0)),
                "mean_correction_ic": float(np.mean([row["correction_ic"] for row in selected])),
            })

    summary_lookup = {(row["window"], row["method"]): row for row in summaries}
    retrieval_above_random = []
    retrieval_above_recent = []
    confidence_all, delta_all = [], []
    for window_name, _, _, _, _ in EVAL_WINDOWS:
        retrieval_above_random.append(
            summary_lookup[(window_name, "retrieval")]["candidate_mean_ic"]
            > summary_lookup[(window_name, "random")]["candidate_mean_ic"]
        )
        retrieval_above_recent.append(
            summary_lookup[(window_name, "retrieval")]["candidate_mean_ic"]
            > summary_lookup[(window_name, "recent")]["candidate_mean_ic"]
        )
        chosen = [row for row in rows if row["window"] == window_name and row["method"] == "retrieval"]
        confidence_all.extend(row["confidence"] for row in chosen)
        delta_all.extend(row["delta"] for row in chosen)

    confidence_audit = loo_confidence_stats(
        np.asarray(confidence_all, dtype=np.float64),
        np.asarray(delta_all, dtype=np.float64),
        int(params["leave_one_block_out_blocks"]),
    )
    hashes_after = {name: sha256(path) for name, path in protected.items()}
    pass_checks = {
        "retrieval_above_random_every_window": bool(all(retrieval_above_random)),
        "retrieval_above_recent_at_least_three_windows": bool(sum(retrieval_above_recent) >= 3),
        "confidence_loo_median_positive": bool(confidence_audit["median"] > 0),
        "confidence_loo_positive_ratio_at_least_0_75": bool(confidence_audit["positive_ratio"] >= 0.75),
        "protected_hashes_unchanged": hashes_before == hashes_after,
    }
    passed = bool(all(pass_checks.values()))

    RESULT.mkdir(parents=True, exist_ok=True)
    with (RESULT / "per_time_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RESULT / "retrieval_neighbors.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["window", "time", "method", "min_library_time", "max_library_time", "library_cutoff_exclusive",
                  "mean_neighbor_distance", "fingerprint_std"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "experiment": "exp_024a_retrieval_diagnostic",
        "decision": "go_to_exp024b" if passed else "stop_retrieval_route",
        "passed": passed,
        "features": feature_names,
        "feature_count": len(feature_names),
        "parameters": params,
        "window_summaries": summaries,
        "retrieval_above_random_by_window": retrieval_above_random,
        "retrieval_above_recent_by_window": retrieval_above_recent,
        "confidence_leave_one_block_out": confidence_audit,
        "pass_checks": pass_checks,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_unchanged": hashes_before == hashes_after,
        "test_arrays_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if passed else "completed_rejected",
        "causal_status": "compliant_diagnostic",
        "baseline_note": "OOF windows use exp016 static strict-OOF proxy; official Valid uses frozen exp021 full prediction",
        "formal_submission_overwritten": False,
        "test_prediction_generated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
