from __future__ import annotations

import numpy as np

from ..config import STOCK_COUNT, TEST_TIME_POINTS


def vector_to_grid(values: np.ndarray, times: np.ndarray, stocks: np.ndarray, split_start: int, time_points: int) -> np.ndarray:
    grid = np.full((time_points, STOCK_COUNT), 0.5, dtype=np.float32)
    grid[np.asarray(times, dtype=np.int32) - int(split_start), np.asarray(stocks, dtype=np.int32)] = values
    return grid


def evaluation_mask(times: np.ndarray, stocks: np.ndarray) -> np.ndarray:
    mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
    mask[np.asarray(times, dtype=np.int32) - 3161, np.asarray(stocks, dtype=np.int32)] = True
    return mask


def validate_prediction(grid: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    report = {
        "shape": list(grid.shape), "dtype": str(grid.dtype), "finite": bool(np.isfinite(grid).all()),
        "evaluation_count": int(mask.sum()), "non_evaluation_count": int((~mask).sum()),
        "non_evaluation_all_0_5": bool(np.all(grid[~mask] == np.float32(0.5))),
        "minimum": float(grid.min()), "maximum": float(grid.max()),
    }
    if grid.shape != (TEST_TIME_POINTS, STOCK_COUNT) or grid.dtype != np.float32 or not report["finite"] or not report["non_evaluation_all_0_5"]:
        raise AssertionError(f"提交预测契约失败：{report}")
    return report
