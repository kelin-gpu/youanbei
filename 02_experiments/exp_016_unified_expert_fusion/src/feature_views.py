from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def robust_rank_features(current: np.ndarray, numeric_count: int = 40) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    raw = current[:, :numeric_count]
    ranks = np.empty_like(raw)
    for col in range(raw.shape[1]):
        ranks[:, col] = rankdata(raw[:, col], method="average") / max(1, raw.shape[0])
    median = np.median(raw, axis=0, keepdims=True)
    iqr = np.quantile(raw, 0.75, axis=0, keepdims=True) - np.quantile(raw, 0.25, axis=0, keepdims=True)
    z = np.clip((raw - median) / np.maximum(iqr, 1e-5), -8.0, 8.0)
    return np.concatenate([raw, ranks.astype(np.float32), z.astype(np.float32)], axis=1)


def state_features(history: np.ndarray, mask: np.ndarray) -> np.ndarray:
    history, mask = np.asarray(history, np.float32), np.asarray(mask, np.float32)
    weight = mask[..., None]
    denom = np.maximum(weight.sum(axis=1), 1.0)
    mean = (history * weight).sum(axis=1) / denom
    centered = history - mean[:, None, :]
    std = np.sqrt((centered * centered * weight).sum(axis=1) / denom + 1e-6)
    recent = history[:, -1, :]
    change = recent - history[:, max(0, history.shape[1] - 21), :]
    coverage = mask.mean(axis=1, keepdims=True)
    return np.concatenate([mean, std, recent, change, coverage], axis=1).astype(np.float32)
