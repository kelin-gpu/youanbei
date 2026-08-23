"""Fill the missing exp021 full-Valid stability audit without touching baselines.

The existing exp021 head/router checkpoints and exp016 full-Valid neural family
cache are reused. Only the official-valid categorical tabular family is fitted
in memory. No model or submission file is overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_experiments"))

from exp_016_unified_expert_fusion.config import (  # noqa: E402
    BASE_WEIGHTS, FAMILIES, MIN_WEIGHTS, RunConfig, STOCK_COUNT, VALID_START, VALID_STOP,
)
from exp_016_unified_expert_fusion.src.data_context import DataContext  # noqa: E402
from exp_016_unified_expert_fusion.src.full_pipeline import (  # noqa: E402
    _head_outputs_by_group, _tabular_arrays,
)
from exp_016_unified_expert_fusion.src.multi_objective_head import MultiObjectiveRankHead  # noqa: E402
from exp_016_unified_expert_fusion.src.ranking import dynamic_blend_family_predictions, rank_ic  # noqa: E402
from exp_016_unified_expert_fusion.src.state_router import StateRouter  # noqa: E402
from exp_016_unified_expert_fusion.src.tabular_experts import (  # noqa: E402
    predict_tabular_family, train_tabular_family,
)
from exp_016_unified_expert_fusion.src.training import restore_torch_checkpoint  # noqa: E402


CACHE = ROOT / "03_cache" / "exp_016_unified_expert_fusion" / "oof_predictions"
EXP021_CACHE = ROOT / "03_cache" / "exp_021_retrain_head_router"
RESULT = ROOT / "04_results" / "_acceptance" / "exp021_validation_audit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def per_group(values: np.ndarray, target: np.ndarray, groups: np.ndarray, times: np.ndarray, drift: np.ndarray):
    rows, offset = [], 0
    for group_index, size_value in enumerate(groups):
        size = int(size_value)
        ic = rank_ic(values[offset:offset + size], target[offset:offset + size])
        rows.append({"time": int(times[offset]), "rank_ic": float(ic), "drift_score": float(drift[group_index])})
        offset += size
    return rows


def main() -> int:
    t0 = time.time()
    protected_paths = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    before = {name: sha256(path) for name, path in protected_paths.items()}
    config = RunConfig(mode="full", training_allowed=True, stage="audit", device="cpu", stock_cap=1024)
    ctx = DataContext(load_sequence=False)

    print("[exp021-audit] fit official-valid categorical tabular in memory", flush=True)
    train_X, train_y, train_rel, train_groups, _ = _tabular_arrays(ctx, "train", 486, VALID_START, config.stock_cap)
    valid_X, valid_y, _, valid_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, STOCK_COUNT)
    models = train_tabular_family(config, train_X, train_y, train_rel, train_groups)
    del train_X, train_y, train_rel, train_groups
    tabular_valid, _ = predict_tabular_family(models, valid_X, valid_groups)
    print("[exp021-audit] tabular prediction complete", flush=True)

    first_six = np.load(CACHE / "full_valid_first_six.npy").astype(np.float32)
    states = torch.from_numpy(np.load(CACHE / "full_valid_states.npy").astype(np.float32))
    groups = np.load(CACHE / "full_valid_group_sizes.npy").astype(np.int64)
    if first_six.shape[0] != tabular_valid.size or int(groups.sum()) != tabular_valid.size:
        raise RuntimeError("full-valid cache and categorical tabular prediction are misaligned")
    first_six[:, 1] = tabular_valid

    offset = 0
    for group_index, size_value in enumerate(groups):
        size = int(size_value)
        states[group_index, 7] = torch.from_numpy(first_six[offset:offset + size]).std(dim=1, unbiased=False).mean()
        offset += size

    head = MultiObjectiveRankHead(expert_count=6, state_dim=8)
    restore_torch_checkpoint(head, EXP021_CACHE / "multi_objective_head.pt")
    base = torch.tensor([BASE_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    minimum = torch.tensor([MIN_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    router = StateRouter(8, len(FAMILIES), base, minimum)
    restore_torch_checkpoint(router, EXP021_CACHE / "state_router.pt")

    experts = torch.from_numpy(first_six)
    groups_t = torch.from_numpy(groups)
    output = _head_outputs_by_group(head, experts, states, groups_t)
    family = {FAMILIES[i]: first_six[:, i] for i in range(6)}
    family["multi_objective_rank"] = output["score"].numpy().astype(np.float32)
    router.eval()
    with torch.no_grad():
        confidence = torch.stack([part.mean() for part in torch.split(output["confidence"], groups.tolist())])
        weights = router(states, confidence).numpy().astype(np.float32)
    prediction = dynamic_blend_family_predictions(family, groups, weights)

    # Drift proxy: mean absolute standardized current feature value across the
    # 40 manifest-selected raw features. It uses Valid X only, never labels.
    drift_scores, offset = [], 0
    for size_value in groups:
        size = int(size_value)
        drift_scores.append(float(np.mean(np.abs(valid_X[offset:offset + size, :40]))))
        offset += size
    drift_scores = np.asarray(drift_scores, dtype=np.float64)
    valid_times = np.asarray(ctx.common["valid"]["time"], dtype=np.int32)
    rows = per_group(prediction, valid_y, groups, valid_times, drift_scores)
    scores = np.asarray([row["rank_ic"] for row in rows], dtype=np.float64)
    finite = np.isfinite(scores)
    threshold = float(np.quantile(drift_scores, 0.8))
    high = drift_scores >= threshold
    worst_count = max(1, int(np.ceil(scores.size * 0.10)))
    worst = np.sort(scores[finite])[:worst_count]

    RESULT.mkdir(parents=True, exist_ok=True)
    np.save(RESULT / "full_valid_prediction.npy", prediction.astype(np.float32))
    np.save(RESULT / "full_valid_dynamic_weights.npy", weights)
    with (RESULT / "per_time_ic.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time", "rank_ic", "drift_score"])
        writer.writeheader()
        writer.writerows(rows)

    after = {name: sha256(path) for name, path in protected_paths.items()}
    metrics = {
        "experiment": "exp_021_retrain_head_router",
        "task": "missing full-valid/per-time/high-drift acceptance audit",
        "rows": int(prediction.size),
        "groups": int(groups.size),
        "mean_rank_ic": float(np.mean(scores[finite])),
        "median_rank_ic": float(np.median(scores[finite])),
        "std_rank_ic": float(np.std(scores[finite])),
        "positive_ratio": float(np.mean(scores[finite] > 0)),
        "worst": float(np.min(scores[finite])),
        "best": float(np.max(scores[finite])),
        "worst_decile_mean": float(np.mean(worst)),
        "late_half_mean": float(np.mean(scores[scores.size // 2:])),
        "drift_proxy": "per-time mean abs of standardized current selected raw features (tree columns 0:40)",
        "high_drift_quantile": 0.8,
        "high_drift_threshold": threshold,
        "high_drift_group_count": int(high.sum()),
        "high_drift_mean_rank_ic": float(np.mean(scores[high])),
        "other_groups_mean_rank_ic": float(np.mean(scores[~high])),
        "high_drift_delta": float(np.mean(scores[high]) - np.mean(scores[~high])),
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_unchanged": before == after,
        "model_training": "one in-memory official-valid tabular family; no model saved",
        "reused": [
            "exp016 full-valid first-six/state/group caches",
            "exp021 multi_objective_head.pt",
            "exp021 state_router.pt"
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "acceptance_audit_completed",
        "causal_status": "compliant_validation_only",
        "formal_submission_overwritten": False,
        "baseline_prediction_overwritten": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0 if metrics["protected_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
