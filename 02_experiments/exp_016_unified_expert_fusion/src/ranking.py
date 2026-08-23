from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import rankdata


def group_rank(values: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.empty_like(values)
    offset = 0
    for size in groups:
        size = int(size)
        block = values[offset:offset + size]
        if size == 0 or not np.isfinite(block).all():
            raise ValueError("分组排名输入包含空组或非有限值。")
        result[offset:offset + size] = rankdata(block, method="average").astype(np.float32) / float(size)
        offset += size
    if offset != values.size:
        raise ValueError("groups 与预测向量长度不匹配。")
    return result


def rank_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    left = np.asarray(prediction, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    left, right = left[finite], right[finite]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def blend_family_predictions(predictions: dict[str, np.ndarray], groups: Sequence[int], weights: dict[str, float]) -> np.ndarray:
    keys = tuple(predictions)
    if set(keys) != set(weights):
        raise ValueError("专家预测与权重键不一致。")
    weighted = np.zeros_like(next(iter(predictions.values())), dtype=np.float32)
    for name in keys:
        weighted += float(weights[name]) * group_rank(predictions[name], groups)
    return group_rank(weighted, groups)


def dynamic_blend_family_predictions(predictions: dict[str, np.ndarray], groups: Sequence[int], weights: np.ndarray) -> np.ndarray:
    """Blend ranked family outputs with one normalized weight vector per group."""
    names = tuple(predictions)
    groups = np.asarray(groups, np.int32)
    weights = np.asarray(weights, np.float32)
    if weights.shape != (groups.size, len(names)) or not np.isfinite(weights).all():
        raise ValueError("Dynamic weight matrix must be [group, family] and finite.")
    if np.any(weights <= 0) or not np.allclose(weights.sum(1), 1.0, atol=1e-5):
        raise ValueError("Every routed group weight vector must be positive and normalized.")
    ranked = {name: group_rank(predictions[name], groups) for name in names}
    blended = np.empty(sum(int(x) for x in groups), dtype=np.float32)
    offset = 0
    for group_index, size in enumerate(groups):
        size = int(size)
        block = np.zeros(size, dtype=np.float32)
        for family_index, name in enumerate(names):
            block += weights[group_index, family_index] * ranked[name][offset:offset + size]
        blended[offset:offset + size] = block
        offset += size
    return group_rank(blended, groups)
