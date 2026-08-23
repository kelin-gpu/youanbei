"""Inventory completed acceptance tests and derive missing lightweight audits.

This command does not train models. It records evidence already present in the
workspace, builds a PSI/RankIC feature registry, and measures prediction
similarity between the frozen stack and best candidate.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "01_analysis" / "outputs"
RESULT = ROOT / "04_results" / "_acceptance"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def feature_registry() -> list[dict[str, Any]]:
    profiles = {row["feature"]: row for row in read_csv(ANALYSIS / "numeric_feature_profile.csv")}
    rankic = read_csv(ANALYSIS / "rankic_summary.csv")
    by_split = {(row["split"], row["feature"]): row for row in rankic}
    output: list[dict[str, Any]] = []
    for feature, profile in profiles.items():
        train = by_split.get(("train_sample", feature), {})
        valid = by_split.get(("valid", feature), {})
        train_ic = number(train.get("mean_rank_ic"))
        valid_ic = number(valid.get("mean_rank_ic"))
        valid_psi = number(profile["valid_psi"])
        max_psi = number(profile["max_psi"])
        sign_agreement = bool(np.isfinite(train_ic) and np.isfinite(valid_ic) and train_ic * valid_ic > 0)
        valid_signal = abs(valid_ic) if np.isfinite(valid_ic) else 0.0
        # Selection classes are based on Train/Valid evidence only. Test PSI is
        # recorded for monitoring, never used to choose an action.
        if valid_psi >= 1.0 or (valid_signal >= 0.01 and not sign_agreement):
            action = "disable_candidate"
        elif valid_psi >= 0.25 and valid_signal >= 0.01 and sign_agreement:
            action = "rank_or_robust_transform"
        elif valid_psi >= 0.10 or not sign_agreement:
            action = "downweight"
        else:
            action = "stable_keep"
        output.append({
            "feature": feature,
            "feature_index": int(profile["feature_index"]),
            "train_mean_rank_ic": train_ic,
            "train_std_rank_ic": number(train.get("std_rank_ic")),
            "train_positive_rate": number(train.get("positive_rate")),
            "valid_mean_rank_ic": valid_ic,
            "valid_std_rank_ic": number(valid.get("std_rank_ic")),
            "valid_positive_rate": number(valid.get("positive_rate")),
            "sign_agreement": sign_agreement,
            "valid_psi": valid_psi,
            "test_psi": float(profile["test_psi"]),
            "max_psi": max_psi,
            "valid_outlier_rate": float(profile["valid_outlier_rate"]),
            "test_outlier_rate": float(profile["test_outlier_rate"]),
            "audit_action": action,
        })
    return sorted(output, key=lambda row: (row["audit_action"], -row["max_psi"], row["feature_index"]))


def prediction_similarity() -> dict[str, Any]:
    left_path = ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy"
    right_path = ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy"
    time = np.load(ROOT / "03_cache" / "processed_data_v1" / "common" / "test_time.npy", mmap_mode="r")
    stock = np.load(ROOT / "03_cache" / "processed_data_v1" / "common" / "test_stock.npy", mmap_mode="r")
    left = np.load(left_path, mmap_mode="r")
    right = np.load(right_path, mmap_mode="r")
    left_values = np.asarray(left[time - 3161, stock], dtype=np.float64)
    right_values = np.asarray(right[time - 3161, stock], dtype=np.float64)
    pearson = float(np.corrcoef(left_values, right_values)[0, 1])
    per_time: list[float] = []
    for t in range(442):
        mask = time == t + 3161
        if int(mask.sum()) > 2:
            per_time.append(float(np.corrcoef(rankdata(left_values[mask]), rankdata(right_values[mask]))[0, 1]))
    return {
        "left": str(left_path.relative_to(ROOT)).replace("\\", "/"),
        "right": str(right_path.relative_to(ROOT)).replace("\\", "/"),
        "left_sha256": sha256(left_path),
        "right_sha256": sha256(right_path),
        "evaluation_rows": int(left_values.size),
        "pearson_all_evaluation_rows": pearson,
        "per_time_rank_correlation_mean": float(np.mean(per_time)),
        "per_time_rank_correlation_min": float(np.min(per_time)),
        "changed_time_sections": int(np.sum(np.asarray(per_time) < 0.999999)),
    }


def main() -> int:
    RESULT.mkdir(parents=True, exist_ok=True)
    features = feature_registry()
    feature_path = RESULT / "drift_feature_registry.csv"
    with feature_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)

    similarity = prediction_similarity()
    (RESULT / "prediction_similarity.json").write_text(json.dumps(similarity, ensure_ascii=False, indent=2), encoding="utf-8")

    exp021_metrics_path = RESULT / "exp021_validation_audit" / "metrics.json"
    exp021_metrics = json.loads(exp021_metrics_path.read_text(encoding="utf-8")) if exp021_metrics_path.exists() else None
    tests = [
        {"test": "data_and_prediction_contract", "status": "done_reused", "evidence": "scripts/project_audit.py; exp016 safety contract"},
        {"test": "raw_feature_quality_and_psi", "status": "done_reused", "evidence": "01_analysis/outputs/numeric_feature_profile.csv"},
        {"test": "single_feature_rankic_stability", "status": "done_reused", "evidence": "01_analysis/outputs/rankic_summary.csv"},
        {"test": "walk_forward_oof", "status": "done_reused", "evidence": "exp016 OOF cache; exp021 metrics_fit.json"},
        {"test": "exp016_full_valid_and_per_time_stability", "status": "done_reused", "evidence": "exp016/full/official_valid_full_report.json; per_time_ic_full.csv"},
        {"test": "family_ablation_and_resource_audit", "status": "done_reused", "evidence": "exp016/full/family_ablation.json; resource.json"},
        {"test": "psi_rankic_feature_registry", "status": "newly_completed", "evidence": "04_results/_acceptance/drift_feature_registry.csv"},
        {"test": "exp021_vs_exp023h_prediction_similarity", "status": "newly_completed", "evidence": "04_results/_acceptance/prediction_similarity.json"},
        {
            "test": "exp021_full_valid_per_time_and_high_drift",
            "status": "newly_completed" if exp021_metrics else "missing_scheduled",
            "evidence": "04_results/_acceptance/exp021_validation_audit/metrics.json" if exp021_metrics else "to be produced by scripts/audit_exp021_validation.py",
        },
    ]
    counts: dict[str, int] = {}
    for row in features:
        counts[row["audit_action"]] = counts.get(row["audit_action"], 0) + 1
    audit = {
        "scope": "innovation-readiness test history",
        "principle": "reuse completed evidence; only execute missing acceptance tests",
        "tests": tests,
        "feature_registry_counts": counts,
        "prediction_similarity": similarity,
        "exp021_validation_summary": exp021_metrics,
        "training_performed": False,
    }
    (RESULT / "test_history_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
