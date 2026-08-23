from __future__ import annotations

"""T1.4：Conservative Router 保守路由（exp_019，离线后处理，零训练）。

用 w = (1-γ)·w_static + γ·w_router 收缩自由路由，并为 exp015_anchor + tabular
保底合计 ≥60%、深度家族单家 ≤10%。w_static 以 BASE_WEIGHTS 为基础重设。

评估用 T0.1b 产出的全量 Valid 七家族预测 + 全量 Valid 动态权重；提交用
exp016 full 的 Test 七家族预测 + Test 动态权重。逐 γ 扫描后选择最优 γ 生成提交。
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(r"D:\google_dl\book\youanbei")
COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
EXP016_FULL = ROOT / "04_results" / "exp_016_unified_expert_fusion" / "full"
RESULT = ROOT / "04_results" / "exp_019_conservative_router"

FAMILIES = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
            "relational_graph", "foundation_representation", "multi_objective_rank")
# anchor+tabular 合计 0.60，深度家族单家 ≤0.10
W_STATIC = np.array([0.40, 0.20, 0.08, 0.06, 0.06, 0.08, 0.12], dtype=np.float64)
TEST_START = 3161
TEST_TIME_POINTS = 442
STOCK_COUNT = 5282
GAMMAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def group_rank(values, groups):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    result = np.empty_like(values)
    offset = 0
    for size in groups:
        size = int(size)
        result[offset:offset + size] = rankdata(values[offset:offset + size], method="average").astype(np.float32) / float(size)
        offset += size
    return result


def rank_ic(prediction, target):
    left = np.asarray(prediction, dtype=np.float64)
    right = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    left, right = left[finite], right[finite]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def dynamic_blend(predictions, groups, weights):
    names = tuple(predictions)
    groups = np.asarray(groups, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.float32)
    ranked = {name: group_rank(predictions[name], groups) for name in names}
    blended = np.empty(int(groups.sum()), dtype=np.float32)
    offset = 0
    for gi, size in enumerate(groups):
        size = int(size)
        block = np.zeros(size, dtype=np.float32)
        for fi, name in enumerate(names):
            block += weights[gi, fi] * ranked[name][offset:offset + size]
        blended[offset:offset + size] = block
        offset += size
    return group_rank(blended, groups)


def conservative_weights(w_router, gamma):
    w = (1.0 - gamma) * W_STATIC + gamma * np.asarray(w_router, dtype=np.float64)
    # 硬地板：anchor(0)+tabular(1) 合计 ≥ 0.60
    a = w[:, 0] + w[:, 1]
    below = a < 0.60
    if below.any():
        ratio_ab = np.where(below, 0.60 / np.maximum(a, 1e-9), 1.0)
        w[below, 0] *= ratio_ab[below]
        w[below, 1] *= ratio_ab[below]
        other_sum = w[:, 2:].sum(axis=1)
        ratio_o = np.where(below, 0.40 / np.maximum(other_sum, 1e-9), 1.0)
        w[below, 2:] *= ratio_o[below][:, None]
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def per_group_ic(pred, target, groups):
    scores, offset = [], 0
    for size in groups:
        size = int(size)
        scores.append(rank_ic(pred[offset:offset + size], target[offset:offset + size]))
        offset += size
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite))


def main():
    t0 = time.time()
    RESULT.mkdir(parents=True, exist_ok=True)

    # 全量 Valid 七家族预测 + 动态权重（来自 T0.1b）
    full_family = np.load(EXP016_FULL / "full_valid_family_predictions.npy")  # (982972, 7)
    valid_w = np.load(EXP016_FULL / "full_valid_dynamic_weights.npy")          # (243, 7)
    valid_y = np.load(COMMON / "valid_y.npy", mmap_mode="r")
    valid_groups = np.load(COMMON / "valid_group_sizes.npy", mmap_mode="r")
    valid_groups_i = np.asarray(valid_groups, dtype=np.int32)
    valid_family = {name: full_family[:, i] for i, name in enumerate(FAMILIES)}

    # Test 七家族预测 + 动态权重（来自 exp016 full）
    test_family = {}
    for name in FAMILIES:
        test_family[name] = np.load(EXP016_FULL / f"family_{name}.npy").astype(np.float32)
    test_w = np.load(EXP016_FULL / "dynamic_weights.npy")  # (442, 7)
    test_time = np.load(COMMON / "test_time.npy", mmap_mode="r")
    test_stock = np.load(COMMON / "test_stock.npy", mmap_mode="r")
    test_groups = np.asarray(np.load(COMMON / "test_group_sizes.npy"), dtype=np.int32)

    base_ic = per_group_ic(dynamic_blend(valid_family, valid_groups_i, valid_w), valid_y, valid_groups_i)
    print(f"[exp019] baseline (gamma=1 router) full-valid IC = {base_ic:.6f}", flush=True)

    rows = []
    for gamma in GAMMAS:
        vw = conservative_weights(valid_w, gamma)
        ic = per_group_ic(dynamic_blend(valid_family, valid_groups_i, vw), valid_y, valid_groups_i)
        anchor_tabular = float((vw[:, 0] + vw[:, 1]).mean())
        max_deep = float(vw[:, 2:6].max())
        rows.append({"gamma": gamma, "full_valid_ic": ic, "delta_vs_router": round(ic - base_ic, 6),
                     "mean_anchor_tabular": anchor_tabular, "max_deep_family": max_deep})
        print(f"[exp019] gamma={gamma}: IC={ic:.6f} (Δ{ic-base_ic:+.6f}), anchor+tabular={anchor_tabular:.3f}, max_deep={max_deep:.3f}", flush=True)

    # 选择全量 Valid 上最优且不劣于 router 的 γ（尽量小，倾向更强约束）
    best = min(rows, key=lambda r: (-r["full_valid_ic"], r["gamma"]))

    # 生成提交：对每个非负 Δ 的 γ 输出一份（最优放 prediction_1）
    candidates = [r for r in rows if r["delta_vs_router"] >= -1e-6] or [best]
    candidates = sorted(candidates, key=lambda r: (-r["full_valid_ic"], r["gamma"]))

    pred_files = []
    for rank_i, r in enumerate(candidates, start=1):
        tw = conservative_weights(test_w, r["gamma"])
        blended = dynamic_blend(test_family, test_groups, tw)
        grid = np.full((TEST_TIME_POINTS, STOCK_COUNT), 0.5, dtype=np.float32)
        grid[np.asarray(test_time, dtype=np.int32) - TEST_START, np.asarray(test_stock, dtype=np.int32)] = blended
        mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
        mask[np.asarray(test_time, dtype=np.int32) - TEST_START, np.asarray(test_stock, dtype=np.int32)] = True
        contract_ok = (grid.shape == (TEST_TIME_POINTS, STOCK_COUNT) and grid.dtype == np.float32
                       and bool(np.isfinite(grid).all()) and bool(np.all(grid[~mask] == np.float32(0.5))) and int(mask.sum()) == 2042538)
        fn = RESULT / f"prediction_{rank_i}.npy"
        np.save(fn, grid)
        pred_files.append({"gamma": r["gamma"], "file": f"prediction_{rank_i}.npy", "contract_ok": contract_ok})
        print(f"[exp019] saved {fn.name} (gamma={r['gamma']}, contract_ok={contract_ok})", flush=True)

    with open(RESULT / "conservative_weights.csv", "w", encoding="utf-8") as f:
        f.write("gamma,full_valid_ic,delta_vs_router,mean_anchor_tabular,max_deep_family\n")
        for r in rows:
            f.write(f"{r['gamma']},{r['full_valid_ic']:.10f},{r['delta_vs_router']},{r['mean_anchor_tabular']:.6f},{r['max_deep_family']:.6f}\n")

    metrics = {"baseline_router_full_valid_ic": base_ic, "w_static": {name: float(W_STATIC[i]) for i, name in enumerate(FAMILIES)},
               "gamma_scan": rows, "best_gamma": best["gamma"], "submissions": pred_files}
    metadata = {"experiment": "exp_019_conservative_router", "task": "T1.4 conservative router",
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "final_submission_overwritten": False,
                "elapsed_s": round(time.time() - t0, 1)}
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp019] DONE in {time.time()-t0:.1f}s", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
