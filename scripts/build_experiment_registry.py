"""Build the machine-readable experiment registry from existing artifacts.

This script is intentionally conservative: missing values are recorded as
``missing`` and are never inferred from neighboring experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "04_results"
FIELDS = [
    "experiment", "artifact", "baseline", "causal_status", "train_range",
    "valid_range", "local_rank_ic", "online_rank_ic", "prediction_path",
    "prediction_sha256", "metrics_path", "metadata_path", "status",
    "decision", "changed_variable",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def scalar_metric(value: Any) -> str:
    return str(value) if isinstance(value, (int, float, str)) else "missing"


def online_scores() -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in sorted((RESULTS / "_decision_log").glob("*.json")):
        data = read_json(path)
        experiment = str(data.get("experiment", ""))
        candidate = data.get("candidate", {})
        score = candidate.get("online_rank_ic") if isinstance(candidate, dict) else None
        if experiment and isinstance(score, (int, float)):
            scores[experiment] = float(score)
    return scores


def decision_for(experiment: str) -> str:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((RESULTS / "_decision_log").glob("*.json")):
        data = read_json(path)
        if data.get("experiment") == experiment:
            matches.append((path, data))
    if not matches:
        return "missing"
    data = matches[-1][1]
    return str(data.get("decision", data.get("final_verdict", "recorded")))


def causal_status(experiment: str, metrics: dict[str, Any]) -> str:
    if experiment == "exp_023a_future_shift":
        return "invalid_future_information_probe"
    if metrics.get("compliance"):
        return "compliant_claim_recorded"
    return "not_assessed"


def rows() -> list[dict[str, str]]:
    scores = online_scores()
    output: list[dict[str, str]] = []
    for directory in sorted(RESULTS.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        experiment = directory.name
        metrics_path = directory / "metrics.json"
        metadata_path = directory / "metadata.json"
        protocol_path = directory / "protocol.json"
        metrics = read_json(metrics_path)
        metadata = read_json(metadata_path)
        protocol = read_json(protocol_path)
        recorded_decision = decision_for(experiment)
        predictions = sorted(directory.rglob("prediction*.npy"))
        if not predictions:
            predictions = [None]
        for prediction in predictions:
            row = {field: "missing" for field in FIELDS}
            row.update({
                "experiment": experiment,
                "artifact": prediction.name if prediction else "missing",
                "baseline": str(protocol.get("baseline_experiment", (
                    "exp021" if experiment in {"exp_020_tabular_categorical", "exp_021_retrain_head_router"}
                    else "exp021+anchor_surgery" if experiment == "exp_023h_ultimate_surgery"
                    else "missing"
                ))),
                "causal_status": str(protocol.get("causal_status", causal_status(experiment, metrics))),
                "train_range": str(protocol.get("train_range", "missing")),
                "valid_range": str(protocol.get("valid_range", "missing")),
                "local_rank_ic": scalar_metric(
                    metrics.get("online_rank_ic", metrics.get("surgery_valid_ic", metrics.get("blend_valid_ic", metrics.get("capped_valid_mean_ic", (metrics.get("fit_reference", {}) or {}).get("capped_valid_mean_ic", "missing")))))
                ),
                "online_rank_ic": scalar_metric(scores.get(experiment, "missing")),
                "metrics_path": str(metrics_path.relative_to(ROOT)).replace("\\", "/") if metrics_path.exists() else "missing",
                "metadata_path": str(metadata_path.relative_to(ROOT)).replace("\\", "/") if metadata_path.exists() else "missing",
                "status": str(metadata.get("status", protocol.get("status", metrics.get("decision", "recorded")))),
                "decision": (
                    recorded_decision
                    if recorded_decision != "missing"
                    else str(metrics.get("decision", "missing"))
                ),
                "changed_variable": str(protocol.get("changed_variable", metrics.get("experiment", metadata.get("task", "missing")))),
            })
            if prediction:
                row["prediction_path"] = str(prediction.relative_to(ROOT)).replace("\\", "/")
                row["prediction_sha256"] = sha256(prediction)
            output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "experiment_registry.csv")
    args = parser.parse_args()
    target = args.output if args.output.is_absolute() else ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows())
    print(json.dumps({"output": str(target), "rows": len(rows())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
