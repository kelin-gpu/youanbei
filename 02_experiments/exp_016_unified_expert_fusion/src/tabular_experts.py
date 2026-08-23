from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from ..config import LABEL_GAIN, RELEVANCE_LEVELS, RunConfig, require_training
from .ranking import group_rank
from .artifacts import atomic_bytes, atomic_text


CATEGORY_INDICES = tuple(range(408, 417))


def validate_relevance(relevance: np.ndarray, groups: np.ndarray) -> dict[str, int]:
    relevance = np.asarray(relevance)
    groups = np.asarray(groups, np.int32)
    if relevance.ndim != 1 or not np.issubdtype(relevance.dtype, np.integer):
        raise ValueError("Ranking relevance must be a one-dimensional integer array.")
    if relevance.size == 0 or int(groups.sum()) != relevance.size or np.any(groups <= 0):
        raise ValueError("Ranking groups must be positive and cover every relevance label.")
    minimum, maximum = int(relevance.min()), int(relevance.max())
    if minimum < 0 or maximum >= RELEVANCE_LEVELS:
        raise ValueError(f"Relevance labels must be in [0,{RELEVANCE_LEVELS - 1}], got [{minimum},{maximum}].")
    if len(LABEL_GAIN) != RELEVANCE_LEVELS or maximum >= len(LABEL_GAIN):
        raise ValueError("LightGBM label_gain does not cover the relevance mapping.")
    return {"minimum": minimum, "maximum": maximum, "levels": RELEVANCE_LEVELS}


def lambdarank_params(**overrides) -> dict[str, object]:
    params: dict[str, object] = {"objective": "lambdarank", "metric": "None", "verbosity": -1,
                                 "label_gain": list(LABEL_GAIN), "lambdarank_truncation_level": 1024}
    params.update(overrides)
    return params


@dataclass
class TabularDryRun:
    rows: int
    features: int
    groups: int
    cat_features: list[int]
    objectives: tuple[str, ...]


def build_tabular_dry_run(X: np.ndarray, relevance: np.ndarray, groups: np.ndarray) -> TabularDryRun:
    X, relevance, groups = np.asarray(X, np.float32), np.asarray(relevance), np.asarray(groups, np.int32)
    if X.ndim != 2 or X.shape[0] != relevance.size or int(groups.sum()) != X.shape[0]:
        raise ValueError("表格专家 dry-run 输入不一致。")
    validate_relevance(relevance.astype(np.int32, copy=False), groups)
    cat_features = [index for index in CATEGORY_INDICES if index < X.shape[1]]
    return TabularDryRun(rows=int(X.shape[0]), features=int(X.shape[1]), groups=int(groups.size), cat_features=cat_features, objectives=("lgbm_lambdarank", "catboost_yetirank", "lgbm_huber", "xgboost_hist"))


def _catboost_pool(X: np.ndarray, relevance: np.ndarray | None, groups: np.ndarray | None):
    """Build a CatBoost pool without declaring float columns categorical.

    FeaturesData places all numeric columns first and the nine manifest-declared
    categorical columns last.  The same transformation is used for inference.
    """
    import catboost as cb

    X = np.asarray(X, np.float32)
    numeric_index = [index for index in range(X.shape[1]) if index not in CATEGORY_INDICES]
    numeric = np.ascontiguousarray(X[:, numeric_index], dtype=np.float32)
    categorical = np.rint(X[:, CATEGORY_INDICES]).astype(np.int64).astype(str).astype(object)
    data = cb.FeaturesData(num_feature_data=numeric, cat_feature_data=categorical)
    group_id = None if groups is None else np.repeat(np.arange(len(groups), dtype=np.int64), groups)
    return cb.Pool(data, label=relevance, group_id=group_id)


def train_tabular_family(config: RunConfig, X: np.ndarray, target: np.ndarray, relevance: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    require_training(config, "异构表格专家训练")
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb
    X = np.asarray(X, np.float32)
    target = np.asarray(target, np.float32)
    relevance = np.asarray(relevance, np.int32)
    groups = np.asarray(groups, np.int32)
    if X.shape[0] != target.size or target.size != relevance.size or int(groups.sum()) != target.size:
        raise ValueError("Tabular training arrays do not share the same row contract.")
    validate_relevance(relevance, groups)
    dataset = lgb.Dataset(X, label=relevance, group=groups, free_raw_data=True, categorical_feature=list(CATEGORY_INDICES))
    lgb_rank = lgb.train(lambdarank_params(), dataset, num_boost_round=16)
    lgb_huber = lgb.train({"objective": "huber", "metric": "None", "verbosity": -1}, lgb.Dataset(X, label=target, categorical_feature=list(CATEGORY_INDICES)), num_boost_round=16)
    cat_pool = _catboost_pool(X, relevance, groups)
    cat = cb.train(cat_pool, {"loss_function": "YetiRank", "iterations": 16, "verbose": False, "allow_writing_files": False})
    xgb_model = xgb.train({"objective": "rank:pairwise", "tree_method": "hist", "verbosity": 0}, xgb.DMatrix(X, label=relevance, group=groups), num_boost_round=16)
    return {"lgbm_rank": lgb_rank, "lgbm_huber": lgb_huber, "catboost": cat, "xgboost": xgb_model}


def predict_tabular_family(models: dict[str, object], X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict every fitted member and return the within-family rank blend."""
    import xgboost as xgb

    required = {"lgbm_rank", "lgbm_huber", "catboost", "xgboost"}
    if set(models) != required:
        raise RuntimeError(f"Incomplete tabular family: {sorted(set(models) ^ required)}")
    X = np.asarray(X, np.float32)
    raw = {
        "lgbm_rank": np.asarray(models["lgbm_rank"].predict(X), np.float32),
        "lgbm_huber": np.asarray(models["lgbm_huber"].predict(X), np.float32),
        "catboost": np.asarray(models["catboost"].predict(_catboost_pool(X, None, None)), np.float32),
        "xgboost": np.asarray(models["xgboost"].predict(xgb.DMatrix(X)), np.float32),
    }
    return rank_tabular_predictions(raw, groups), raw


def save_tabular_family(models: dict[str, object], directory: Path) -> dict[str, str]:
    """Atomically save native models without passing Unicode paths to C++ libraries."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "lgbm_rank": directory / "lgbm_rank.txt", "lgbm_huber": directory / "lgbm_huber.txt",
        "catboost": directory / "catboost.cbm", "xgboost": directory / "xgboost.ubj",
    }
    atomic_text(paths["lgbm_rank"], models["lgbm_rank"].model_to_string())
    atomic_text(paths["lgbm_huber"], models["lgbm_huber"].model_to_string())
    # CatBoost 1.2 on Windows cannot write directly through a non-ASCII path.
    # Let the native library use an ASCII system temp path, then move bytes with Python.
    with tempfile.TemporaryDirectory(prefix="exp016_catboost_") as temporary:
        native_path = Path(temporary) / "model.cbm"
        models["catboost"].save_model(str(native_path))
        atomic_bytes(paths["catboost"], native_path.read_bytes())
    atomic_bytes(paths["xgboost"], bytes(models["xgboost"].save_raw(raw_format="ubj")))
    return {name: str(path) for name, path in paths.items()}


def load_tabular_family(directory: Path) -> dict[str, object]:
    """Load models saved by save_tabular_family, including from Unicode paths."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    directory = Path(directory)
    paths = {"lgbm_rank": directory / "lgbm_rank.txt", "lgbm_huber": directory / "lgbm_huber.txt",
             "catboost": directory / "catboost.cbm", "xgboost": directory / "xgboost.ubj"}
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete tabular checkpoint: {missing}")
    rank = lgb.Booster(model_str=paths["lgbm_rank"].read_text(encoding="utf-8"))
    huber = lgb.Booster(model_str=paths["lgbm_huber"].read_text(encoding="utf-8"))
    cat = cb.CatBoost()
    with tempfile.TemporaryDirectory(prefix="exp016_catboost_load_") as temporary:
        native_path = Path(temporary) / "model.cbm"
        native_path.write_bytes(paths["catboost"].read_bytes())
        cat.load_model(str(native_path))
    xgb_model = xgb.Booster(); xgb_model.load_model(bytearray(paths["xgboost"].read_bytes()))
    return {"lgbm_rank": rank, "lgbm_huber": huber, "catboost": cat, "xgboost": xgb_model}


def rank_tabular_predictions(predictions: dict[str, np.ndarray], groups: np.ndarray) -> np.ndarray:
    return np.mean(np.stack([group_rank(value, groups) for value in predictions.values()]), axis=0).astype(np.float32)
