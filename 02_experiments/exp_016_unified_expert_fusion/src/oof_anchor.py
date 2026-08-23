from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import EXP015_DIR, RunConfig, TEST_START, require_training
from .data_context import DataContext
from .artifacts import atomic_text
from .tabular_experts import lambdarank_params, validate_relevance


def load_exp015_test_anchor(ctx: DataContext) -> np.ndarray:
    path = EXP015_DIR / "prediction.npy"
    if not path.exists():
        raise FileNotFoundError(f"缺少 exp015 锚点：{path}")
    grid = np.load(path, mmap_mode="r", allow_pickle=False)
    if grid.shape != (442, 5282) or grid.dtype != np.float32 or not np.isfinite(grid).all():
        raise RuntimeError("exp015 锚点预测契约不匹配。")
    times = np.asarray(ctx.common["test"]["time"], dtype=np.int32) - TEST_START
    stocks = np.asarray(ctx.common["test"]["stock"], dtype=np.int32)
    return np.asarray(grid[times, stocks], dtype=np.float32)


def anchor_oof_contract(ctx: DataContext) -> dict[str, object]:
    """明确 full 阶段的 OOF 锚点不能回退到同段拟合预测。"""
    return {
        "source": "exp009/exp015 walk-forward reconstruction",
        "folds": ["fold_1", "fold_2", "fold_3"],
        "requires_train_only_models": True,
        "test_anchor_available": bool((EXP015_DIR / "prediction.npy").exists()),
        "test_labels_loaded": "y" in ctx.common["test"],
    }


def train_exp015_anchor(config: RunConfig, X: np.ndarray, relevance: np.ndarray, groups: np.ndarray):
    """Reconstruct a strict walk-forward exp015-style rank anchor.

    Existing exp015 Test predictions are reusable, but a prediction trained on
    the same label period is never used as OOF.  Therefore each exp016 fold
    rebuilds the established robust rank anchor from that fold's earlier rows.
    """
    require_training(config, "strict exp015 OOF anchor reconstruction")
    import lightgbm as lgb
    relevance = np.asarray(relevance, np.int32); groups = np.asarray(groups, np.int32)
    validate_relevance(relevance, groups)
    dataset = lgb.Dataset(np.asarray(X, np.float32), label=relevance, group=groups, free_raw_data=True)
    return lgb.train(lambdarank_params(learning_rate=0.05, num_leaves=31, seed=42), dataset, num_boost_round=32)


def predict_exp015_anchor(model, X: np.ndarray) -> np.ndarray:
    values = np.asarray(model.predict(np.asarray(X, np.float32)), dtype=np.float32)
    if values.ndim != 1 or values.size != X.shape[0] or not np.isfinite(values).all():
        raise RuntimeError("Strict exp015 anchor prediction contract failed.")
    return values


def save_exp015_anchor(model, path: Path) -> None:
    atomic_text(Path(path), model.model_to_string())


def load_exp015_anchor(path: Path):
    import lightgbm as lgb
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return lgb.Booster(model_str=path.read_text(encoding="utf-8"))
