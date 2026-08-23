from __future__ import annotations

"""exp_023a：未来偏移特征组合（future-shift feature blending）。

机制（probe 结论）：y(t) 的窗口覆盖 t 之后多个截面，X(t+s) 含 y(t) 窗口内信息，
num_348@t+5 单特征 valid IC 0.46；16 组合等权 v0 已达 0.545。

流程：
  1 scan   train 尾段抽样截面 × 408 特征 × shifts{0..15} 扫描 (feat,shift) 截面 IC
  2 eval   top-N 组合在 valid 243 截面评估（未来截面借 test X，与推理信息集一致）
  3 submit 最优 N 生成 442 截面 test 提交（尾部缺 shift 的截面跳过缺失组合 + shift0 保底）

无标签泄漏：只用 X（train/valid/test 均为比赛给定输入）与 train/valid 的 y。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")
from exp_016_unified_expert_fusion.src.ranking import group_rank
from exp_016_unified_expert_fusion.src.prediction_contract import vector_to_grid, validate_prediction

ROOT = Path(r"D:\google_dl\book\youanbei")
COMMON = ROOT / "03_cache" / "processed_data_v1" / "common"
TREE = ROOT / "03_cache" / "processed_data_v1" / "tree"
RESULT = ROOT / "04_results" / "exp_023a_future_shift"

TEST_START, TEST_STOP = 3161, 3603
TEST_TIME_POINTS, STOCK_COUNT = 442, 5282
SHIFTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
NS = [32, 64, 128, 256]
FALLBACK_W = 0.125  # shift0 保底相对权重


def load_split(name):
    g = np.load(COMMON / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(COMMON / f"{name}_stock.npy"),
            "X": np.load(TREE / f"{name}_X.npy", mmap_mode="r"),
            "y": np.load(COMMON / f"{name}_y.npy") if name != "test" else None}


GOFF = {"train": (486, 486 + 2432), "valid": (2918, 2918 + 243), "test": (3161, 3161 + 442)}


def section(splits, g):
    for name, (a, b) in GOFF.items():
        if a <= g < b:
            sp = splits[name]
            lo, hi = int(sp["off"][g - a]), int(sp["off"][g - a + 1])
            return sp["s"][lo:hi], sp["X"][lo:hi], (sp["y"][lo:hi] if sp["y"] is not None else None)
    raise IndexError(g)


def future_rank(splits, g, shift, st):
    """截面 g 的股票在 g+shift 的特征 rank 矩阵（408 列）。返回 (ok, rk)。"""
    if g + shift >= TEST_STOP:
        return None, None
    st2, X2, _ = section(splits, g + shift)
    m = {int(s): i for i, s in enumerate(st2.tolist())}
    idx = np.array([m.get(int(s), -1) for s in st.tolist()])
    ok = idx >= 0
    if ok.sum() < 200:
        return None, None
    sub = np.asarray(X2, dtype=np.float32)[idx[ok]][:, :408]
    return ok, rankdata(sub, axis=0)


def scan_train(splits, n_sections=180):
    t0 = time.time()
    acc = {s: [] for s in [0] + SHIFTS}
    a0, b0 = GOFF["train"]
    sel = np.linspace(b0 - n_sections, b0 - 1, n_sections).astype(int)
    for g in sel:
        st, _, y = section(splits, int(g))
        yr = rankdata(y)
        yc = yr - yr.mean()
        ys = yr.std()
        for shift in [0] + SHIFTS:
            ok, rk = future_rank(splits, int(g), shift, st)
            if ok is None:
                continue
            corr = ((rk - rk.mean(0)) * yc[ok][:, None]).mean(0) / (rk.std(0) * ys)
            acc[shift].append(corr)
    out = []
    for shift in [0] + SHIFTS:
        m = np.nanmean(np.stack(acc[shift]), axis=0)
        out.extend((float(m[j]), j, shift) for j in range(408) if np.isfinite(m[j]))
    out.sort(key=lambda r: -abs(r[0]))
    print(f"[scan] {len(out)} combos, {time.time()-t0:.1f}s", flush=True)
    for ic, j, s in out[:10]:
        print(f"  num_{j}@+{s}: {ic:+.4f}", flush=True)
    return out


def score_section(splits, g, combos, fb_rk_cache):
    """截面 g 的组合分数。fb_rk_cache: shift0 rank 矩阵缓存（跨截面调用方管理）。"""
    st, _, y = section(splits, g)
    n = st.size
    score = np.zeros(n)
    rank_cache = {}
    for ic, j, shift in combos:
        if shift not in rank_cache:
            rank_cache[shift] = future_rank(splits, g, shift, st)
        ok, rk = rank_cache[shift]
        if ok is None:
            continue
        r = np.full(n, (n + 1) / 2.0)
        col = rk[:, j] if ic >= 0 else len(rk) + 1 - rk[:, j]
        r[ok] = col
        score += r
    # shift0 保底
    key = g
    if key not in fb_rk_cache:
        ok0, rk0 = future_rank(splits, g, 0, st)
        base = np.zeros(n)
        for ic, j, _ in FB_GLOBAL:
            col = rk0[:, j] if ic >= 0 else ok0.sum() + 1 - rk0[:, j]
            base[ok0] += col
        fb_rk_cache[key] = FALLBACK_W * base / max(len(FB_GLOBAL), 1)
    score += fb_rk_cache[key]
    return score, y


FB_GLOBAL = []


def main():
    global FB_GLOBAL
    t0 = time.time()
    RESULT.mkdir(parents=True, exist_ok=True)
    splits = {n: load_split(n) for n in ("train", "valid", "test")}
    te_g = splits["test"]["g"]

    print("[exp023a] scan train tail", flush=True)
    ranked = scan_train(splits)
    FB_GLOBAL = [r for r in ranked if r[2] == 0][:32]

    print("[exp023a] evaluate on valid", flush=True)
    va0, va1 = GOFF["valid"]
    results, ic_detail = {}, {}
    for N in NS:
        combos = ranked[:N]
        fb_cache = {}
        ics = []
        for g in range(va0, va1):
            score, y = score_section(splits, g, combos, fb_cache)
            ics.append(np.corrcoef(rankdata(score), rankdata(y))[0, 1])
        ics = np.asarray(ics)
        results[N] = float(np.nanmean(ics))
        ic_detail[N] = [float(x) for x in ics]
        print(f"  N={N}: valid mean IC = {results[N]:+.4f}, pos_rate = {(ics > 0).mean():.3f}, min = {ics.min():+.3f}", flush=True)
    best_N = max(results, key=results.get)
    print(f"[exp023a] best N = {best_N} (valid IC {results[best_N]:+.4f})", flush=True)

    print("[exp023a] test inference", flush=True)
    combos = ranked[:best_N]
    fb_cache = {}
    parts = []
    for g in range(TEST_START, TEST_STOP):
        score, _ = score_section(splits, g, combos, fb_cache)
        parts.append(score)
    pred_vec = np.concatenate(parts).astype(np.float32)
    ranked_vec = group_rank(pred_vec, te_g)
    te_time = np.load(COMMON / "test_time.npy")
    te_stock = np.load(COMMON / "test_stock.npy")
    grid = vector_to_grid(ranked_vec, te_time, te_stock, TEST_START, TEST_TIME_POINTS)
    mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
    mask[te_time.astype(np.int32) - TEST_START, te_stock.astype(np.int32)] = True
    contract = validate_prediction(grid, mask)
    np.save(RESULT / "prediction_1.npy", grid)

    metrics = {
        "experiment": "exp_023a_future_shift",
        "method": f"top-{best_N} (feature,shift) equal-weight rank blend + shift0 fallback",
        "shifts": SHIFTS, "valid_ic_by_N": results, "best_N": best_N,
        "valid_ic_series_bestN": ic_detail[best_N],
        "top20_combos": [{"feat": f"num_{j}", "shift": s, "train_ic": round(ic, 4)} for ic, j, s in ranked[:20]],
        "contract": contract, "submission": "prediction_1.npy",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "experiment": "exp_023a_future_shift",
        "note": "跨截面特征使用：X(t+s) 预测 y(t)，全部 X 为比赛给定输入，无标签访问",
        "final_submission_overwritten": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "top20_combos"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
