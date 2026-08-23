from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "data.z").exists() and (parent / "03_cache").exists():
            return parent
    raise RuntimeError("无法定位项目根目录（需要 data.z 与 03_cache）。")


ROOT: Final = find_project_root()
EXPERIMENT_ID: Final = "exp_016_unified_expert_fusion"
DATASET_DIR: Final = ROOT / "03_cache" / "processed_data_v1"
CACHE_DIR: Final = ROOT / "03_cache" / EXPERIMENT_ID
RESULT_DIR: Final = ROOT / "04_results" / EXPERIMENT_ID
EXP015_DIR: Final = ROOT / "04_results" / "exp_015_drift_robust_rank" / "integrated_v2" / "full"
FINAL_SUBMISSION: Final = ROOT / "04_results" / "final_submission" / "prediction.npy"

TIME_COUNT: Final = 3603
STOCK_COUNT: Final = 5282
TREE_FEATURES: Final = 419
SEQUENCE_FEATURES: Final = 40
RELEVANCE_LEVELS: Final = 64
LABEL_GAIN: Final = tuple(range(RELEVANCE_LEVELS))
TRAIN_START, TRAIN_STOP = 486, 2918
VALID_START, VALID_STOP = 2918, 3161
TEST_START, TEST_STOP = 3161, 3603
TEST_TIME_POINTS: Final = TEST_STOP - TEST_START
HISTORY_SCALES: Final = (20, 60, 240)
PROTOTYPE_COUNT: Final = 32
NEIGHBOR_COUNT: Final = 16
SEEDS: Final = (42, 2026, 3407)

FAMILIES: Final = (
    "exp015_anchor", "tabular", "dual_axis", "time_frequency",
    "relational_graph", "foundation_representation", "multi_objective_rank",
)
BASE_WEIGHTS: Final = {
    "exp015_anchor": 0.20, "tabular": 0.15, "dual_axis": 0.18,
    "time_frequency": 0.10, "relational_graph": 0.10,
    "foundation_representation": 0.10, "multi_objective_rank": 0.17,
}
MIN_WEIGHTS: Final = {
    "exp015_anchor": 0.10, "tabular": 0.08, "dual_axis": 0.10,
    "time_frequency": 0.05, "relational_graph": 0.05,
    "foundation_representation": 0.05, "multi_objective_rank": 0.08,
}
OOF_FOLDS: Final = (
    ("fold_1", 486, 1459, 1459, 1945),
    ("fold_2", 486, 1945, 1945, 2432),
    ("fold_3", 486, 2432, 2432, 2918),
)


@dataclass(frozen=True)
class RunConfig:
    mode: str
    training_allowed: bool
    stage: str
    device: str
    stock_cap: int = 1024
    real_preflight_times: int = 4

    @property
    def run_dir(self) -> Path:
        return RESULT_DIR / self.mode

    @classmethod
    def from_environment(cls) -> "RunConfig":
        mode = os.environ.get("DSCR_EXP016_MODE", "smoke").strip().lower()
        if mode not in {"static", "smoke", "preflight", "full"}:
            raise ValueError("DSCR_EXP016_MODE 仅允许 static、smoke、preflight、full。")
        stage = os.environ.get("DSCR_EXP016_STAGE", "all").strip().lower()
        if stage not in {"all", "pretrain", "experts", "head", "router", "final"}:
            raise ValueError("DSCR_EXP016_STAGE 不合法。")
        allowed = os.environ.get("DSCR_EXP016_ALLOW_TRAINING", "").strip() == "YES"
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        return cls(mode=mode, training_allowed=allowed, stage=stage, device=device)


def require_training(config: RunConfig, operation: str) -> None:
    if config.mode != "full" or not config.training_allowed:
        raise PermissionError(
            f"拒绝 {operation}：真实训练仅允许 DSCR_EXP016_MODE=full 且 "
            "DSCR_EXP016_ALLOW_TRAINING=YES。"
        )
