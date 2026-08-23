from __future__ import annotations

"""Forecast exp023a's future features causally and test their Valid value.

Only Train and Valid arrays are opened.  Future feature values are supervised
targets during historical fitting/evaluation, never inference inputs.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
TREE = ROOT / "03_cache" / "processed_data_v1" / "tree"
SOURCE_METRICS = ROOT / "04_results" / "exp_023a_future_shift" / "metrics.json"
VALID_BASE_PATH = ROOT / "04_results" / "_acceptance" / "exp021_validation_audit" / "full_valid_prediction.npy"
RESULT = ROOT / "04_results" / "exp_026a_causal_future_feature_proxy"
PROTOCOL = RESULT / "protocol.json"

TRAIN_START, TRAIN_STOP = 486, 2918
VALID_START, VALID_STOP = 2918, 3161
STOCK_COUNT = 5282


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_rank_ic(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 200:
        return 0.0
    lr = rankdata(left[finite], method="average")
    rr = rankdata(right[finite], method="average")
    return float(np.corrcoef(lr, rr)[0, 1])


def dense_labels() -> np.ndarray:
    grid = np.full((VALID_STOP - VALID_START, STOCK_COUNT), np.nan, dtype=np.float32)
    times = np.load(COMMON / "valid_time.npy", mmap_mode="r")
    stocks = np.load(COMMON / "valid_stock.npy", mmap_mode="r")
    values = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    grid[np.asarray(times, dtype=np.int32) - VALID_START, np.asarray(stocks, dtype=np.int32)] = values
    return grid


def dense_valid_base() -> np.ndarray:
    grid = np.full((VALID_STOP - VALID_START, STOCK_COUNT), np.nan, dtype=np.float32)
    times = np.load(COMMON / "valid_time.npy", mmap_mode="r")
    stocks = np.load(COMMON / "valid_stock.npy", mmap_mode="r")
    values = np.load(VALID_BASE_PATH, mmap_mode="r")
    grid[np.asarray(times, dtype=np.int32) - VALID_START, np.asarray(stocks, dtype=np.int32)] = values
    return grid


def feature_grid(feature_index: int) -> np.ndarray:
    grid = np.full((VALID_STOP - TRAIN_START, STOCK_COUNT), np.nan, dtype=np.float32)
    for split, start in (("train", TRAIN_START), ("valid", VALID_START)):
        times = np.load(COMMON / f"{split}_time.npy", mmap_mode="r")
        stocks = np.load(COMMON / f"{split}_stock.npy", mmap_mode="r")
        tree = np.load(TREE / f"{split}_X.npy", mmap_mode="r")
        values = np.asarray(tree[:, feature_index], dtype=np.float32)
        grid[np.asarray(times, dtype=np.int32) - TRAIN_START, np.asarray(stocks, dtype=np.int32)] = values
    return grid


def training_arrays(grid: np.ndarray, shift: int, lags: list[int], cap: int) -> tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = [], []
    first = TRAIN_START + max(lags)
    last = TRAIN_STOP - shift
    for time_index in range(first, last):
        rows = [grid[time_index - lag - TRAIN_START] for lag in lags]
        target = grid[time_index + shift - TRAIN_START]
        matrix = np.stack(rows, axis=1)
        finite = np.isfinite(target) & np.isfinite(matrix).all(axis=1)
        available = np.flatnonzero(finite)
        if available.size == 0:
            continue
        selected = available[np.linspace(0, available.size - 1, min(cap, available.size), dtype=np.int64)]
        x_parts.append(matrix[selected])
        y_parts.append(target[selected])
    return np.concatenate(x_parts).astype(np.float32), np.concatenate(y_parts).astype(np.float32)


def valid_forecast(model, grid: np.ndarray, query_times: np.ndarray, lags: list[int]) -> np.ndarray:
    output = np.full((query_times.size, STOCK_COUNT), np.nan, dtype=np.float32)
    for index, time_index in enumerate(query_times):
        matrix = np.stack([grid[time_index - lag - TRAIN_START] for lag in lags], axis=1)
        finite = np.isfinite(matrix).all(axis=1)
        output[index, finite] = model.predict(matrix[finite]).astype(np.float32)
    return output


def add_rank_signal(accumulator: np.ndarray, counts: np.ndarray, values: np.ndarray, y: np.ndarray) -> None:
    for time_index in range(values.shape[0]):
        finite = np.isfinite(values[time_index]) & np.isfinite(y[time_index])
        if int(finite.sum()) < 200:
            continue
        ranks = rankdata(values[time_index, finite], method="average")
        ranks = (ranks - 1.0) / max(1, ranks.size - 1)
        accumulator[time_index, finite] += ranks
        counts[time_index, finite] += 1


def candidate_metrics(base: np.ndarray, y: np.ndarray, signal_sum: np.ndarray,
                      signal_count: np.ndarray, alpha: float) -> dict[str, object]:
    base_ics, candidate_ics = [], []
    for time_index in range(y.shape[0]):
        finite = np.isfinite(base[time_index]) & np.isfinite(y[time_index]) & (signal_count[time_index] > 0)
        base_rank = rankdata(base[time_index, finite], method="average")
        signal = signal_sum[time_index, finite] / signal_count[time_index, finite]
        candidate = (1.0 - alpha) * base_rank + alpha * rankdata(signal, method="average")
        target = y[time_index, finite]
        base_ics.append(safe_rank_ic(base_rank, target))
        candidate_ics.append(safe_rank_ic(candidate, target))
    base_arr = np.asarray(base_ics, dtype=np.float64)
    candidate_arr = np.asarray(candidate_ics, dtype=np.float64)
    delta = candidate_arr - base_arr
    thirds = np.array_split(np.arange(delta.size), 3)
    return {
        "sections": int(delta.size),
        "baseline_mean_ic": float(np.mean(base_arr)),
        "candidate_mean_ic": float(np.mean(candidate_arr)),
        "mean_delta": float(np.mean(delta)),
        "positive_delta_ratio": float(np.mean(delta > 0)),
        "third_deltas": [float(np.mean(delta[index])) for index in thirds],
        "worst_delta": float(np.min(delta)),
    }


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    params = protocol["preregistered_parameters"]
    lags = [int(value) for value in params["history_lags"]]
    cap = int(params["train_stock_cap_per_time"])
    alpha = float(params["candidate_shrink_alpha"])
    query_stop = int(params["common_valid_stop"])
    query_times = np.arange(VALID_START, query_stop, dtype=np.int32)
    combo_source = json.loads(SOURCE_METRICS.read_text(encoding="utf-8"))["top20_combos"]
    combos = [(int(row["feat"].split("_")[1]), int(row["shift"])) for row in combo_source]

    protected = {
        "exp021_prediction": ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy",
        "exp023h_prediction": ROOT / "04_results" / "exp_023h_ultimate_surgery" / "prediction_1.npy",
        "formal_submission": ROOT / "04_results" / "final_submission" / "prediction.npy",
    }
    hashes_before = {name: sha256(path) for name, path in protected.items()}
    y = dense_labels()[:query_times.size]
    base = dense_valid_base()[:query_times.size]
    forecast_sum = np.zeros_like(y, dtype=np.float64)
    forecast_count = np.zeros_like(y, dtype=np.int16)
    persistence_sum = np.zeros_like(y, dtype=np.float64)
    persistence_count = np.zeros_like(y, dtype=np.int16)
    combo_metrics = []

    by_feature: dict[int, list[int]] = {}
    for feature, shift in combos:
        by_feature.setdefault(feature, []).append(shift)
    for feature, shifts in by_feature.items():
        print(f"feature={feature}, shifts={sorted(shifts)}", flush=True)
        grid = feature_grid(feature)
        for shift in shifts:
            train_x, train_target = training_arrays(grid, shift, lags, cap)
            model = make_pipeline(StandardScaler(), Ridge(alpha=float(params["ridge_alpha"])))
            model.fit(train_x, train_target)
            prediction = valid_forecast(model, grid, query_times, lags)
            persistence = np.stack([grid[time_index - TRAIN_START] for time_index in query_times])
            add_rank_signal(forecast_sum, forecast_count, prediction, y)
            add_rank_signal(persistence_sum, persistence_count, persistence, y)
            forecast_ics, persistence_ics, actual_label_ics = [], [], []
            for local_index, time_index in enumerate(query_times):
                actual_future = grid[time_index + shift - TRAIN_START]
                forecast_ics.append(safe_rank_ic(prediction[local_index], actual_future))
                persistence_ics.append(safe_rank_ic(persistence[local_index], actual_future))
                actual_label_ics.append(safe_rank_ic(actual_future, y[local_index]))
            combo_metrics.append({
                "feature": feature,
                "shift": shift,
                "train_rows": int(train_target.size),
                "forecast_future_feature_rank_ic": float(np.mean(forecast_ics)),
                "persistence_future_feature_rank_ic": float(np.mean(persistence_ics)),
                "actual_future_feature_to_label_rank_ic": float(np.mean(actual_label_ics)),
            })
            print(f"  shift={shift}: forecast IC={np.mean(forecast_ics):+.4f}, persistence={np.mean(persistence_ics):+.4f}", flush=True)
        del grid

    # Compare forecast and persistence on the exact same stock-time mask.
    common_signal = (forecast_count > 0) & (persistence_count > 0)
    forecast_common_count = np.where(common_signal, forecast_count, 0)
    persistence_common_count = np.where(common_signal, persistence_count, 0)
    forecast_metrics = candidate_metrics(base, y, forecast_sum, forecast_common_count, alpha)
    persistence_metrics = candidate_metrics(base, y, persistence_sum, persistence_common_count, alpha)
    hashes_after = {name: sha256(path) for name, path in protected.items()}
    pass_checks = {
        "forecast_delta_at_least_0_0025": forecast_metrics["mean_delta"] >= 0.0025,
        "forecast_above_persistence_by_0_0005": (
            forecast_metrics["candidate_mean_ic"] - persistence_metrics["candidate_mean_ic"] >= 0.0005
        ),
        "all_temporal_third_deltas_positive": all(value > 0 for value in forecast_metrics["third_deltas"]),
        "protected_hashes_unchanged": hashes_before == hashes_after,
    }
    passed = bool(all(pass_checks.values()))
    metrics = {
        "experiment": "exp_026a_causal_future_feature_proxy",
        "decision": "go_to_exp026b" if passed else "stop_causal_future_proxy_route",
        "passed": passed,
        "parameters": params,
        "combos": combo_metrics,
        "forecast_candidate": forecast_metrics,
        "persistence_candidate": persistence_metrics,
        "pass_checks": pass_checks,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_unchanged": hashes_before == hashes_after,
        "test_arrays_loaded": False,
        "prediction_generated": False,
        "online_submission_used": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "status": "completed_go" if passed else "completed_rejected",
        "causal_status": "compliant_train_future_target_only",
        "formal_submission_overwritten": False,
        "test_prediction_generated": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
