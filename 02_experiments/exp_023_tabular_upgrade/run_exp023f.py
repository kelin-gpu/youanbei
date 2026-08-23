from __future__ import annotations

"""exp_023f：多 lag 锚点递归 + 细化衰减曲线（冲击 0.12 的最后一枪）。

新结构：test 第 k 截面的 lag_k = 真实 y(3160)（前 6 截面拥有多条真实锚链）。
mB_multi 输入 [X(t), lag1..lag6 rank]，7..K 截面的锚链退化为自有预测。
valid 同构选择 K/w0/gamma，手术只改前 K 截面，其余与 exp021 一致。
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")
from exp_016_unified_expert_fusion.src.ranking import group_rank, dynamic_blend_family_predictions
from exp_016_unified_expert_fusion.src.prediction_contract import vector_to_grid, validate_prediction

ROOT = Path(r"D:\google_dl\book\youanbei")
C = ROOT / "03_cache" / "processed_data_v1" / "common"
T = ROOT / "03_cache" / "processed_data_v1" / "tree"
F16 = ROOT / "04_results" / "exp_016_unified_expert_fusion" / "full"
P021 = ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy"
RESULT = ROOT / "04_results" / "exp_023f_multilag_surgery"
RESULT.mkdir(parents=True, exist_ok=True)

TEST_START, TEST_TIME_POINTS, STOCK_COUNT = 3161, 442, 5282
FAMILIES = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
            "relational_graph", "foundation_representation", "multi_objective_rank")
CAP = 768
ROUNDS = 80
ALPHA = 0.3
LAGS = (1, 2, 3, 4, 5, 6)
PARAMS = dict(objective="huber", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1)


def load(name):
    g = np.load(C / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(C / f"{name}_stock.npy"),
            "y": np.load(C / f"{name}_y.npy") if name != "test" else None,
            "X": np.load(T / f"{name}_X.npy", mmap_mode="r")}


def lag_rank(src, st, n):
    r = np.full(n, 0.5, dtype=np.float32)
    vals = np.array([src.get(int(s), np.nan) for s in st])
    ok = np.isfinite(vals)
    if ok.sum() > 10:
        r[ok] = ((rankdata(vals[ok]) - 1) / (ok.sum() - 1)).astype(np.float32)
    return r


def sec_dicts(sp, i):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    return dict(zip(sp["s"][a:b].tolist(), sp["y"][a:b].tolist()))


t0 = time.time()
tr, va, te = load("train"), load("valid"), load("test")
n_tr = len(tr["g"])

rowsA, rowsB, ys = [], [], []
hist_tr = [sec_dicts(tr, i) for i in range(n_tr)]
for i in range(6, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    lagcols = [lag_rank(hist_tr[i - k], st, st.size) for k in LAGS]
    rowsA.append(sub)
    rowsB.append(np.column_stack([sub] + lagcols))
    ys.append(tr["y"][a:b][take])
XA, XB, yv = np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)
del rowsA, rowsB
mA = lgb.train(PARAMS, lgb.Dataset(XA, label=yv), num_boost_round=ROUNDS)
mB = lgb.train(PARAMS, lgb.Dataset(XB, label=yv), num_boost_round=ROUNDS)
del XA, XB
print(f"models done {time.time()-t0:.0f}s", flush=True)


def recurse_multi(sp, past, alpha_hi=0.85, alpha_lo=0.3, hi_n=6):
    """alpha 前置调度：前 hi_n 截面锚链真实度高用 alpha_hi，之后退到 alpha_lo。"""
    outs = []
    past = list(past)
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        n = st.size
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        pA = mA.predict(sub)
        lagcols = [lag_rank(past[-k] if len(past) >= k else {}, st, n) for k in LAGS]
        pL = mB.predict(np.column_stack([sub] + lagcols))
        al = alpha_hi if i < hi_n else alpha_lo
        p = (1 - al) * pA + al * pL
        outs.append(p)
        past.append(dict(zip(st.tolist(), p.tolist())))
    return outs


rec_va_parts = recurse_multi(va, hist_tr[-6:])
hist_va = [sec_dicts(va, i) for i in range(len(va["g"]))]
rec_te_parts = recurse_multi(te, hist_va[-6:])
print(f"recursion done {time.time()-t0:.0f}s", flush=True)

# 分量与栈
fam = np.load(F16 / "full_valid_family_predictions.npy")
wts = np.load(F16 / "full_valid_dynamic_weights.npy")
stack_va = dynamic_blend_family_predictions({n: fam[:, k] for k, n in enumerate(FAMILIES)}, va["g"], wts)
stack_va_parts = list(np.split(stack_va, np.cumsum(va["g"])[:-1]))
grid021 = np.load(P021)
te_time = np.load(C / "test_time.npy").astype(np.int32)
te_stock = np.load(C / "test_stock.npy").astype(np.int32)
stack_te_parts = list(np.split(grid021[te_time - TEST_START, te_stock], np.cumsum(te["g"])[:-1]))


def ics(parts):
    out, off = [], 0
    for i, p in enumerate(parts):
        n = int(va["g"][i])
        out.append(np.corrcoef(rankdata(p), rankdata(va["y"][off:off + n]))[0, 1])
        off += n
    return np.asarray(out)


base = float(ics(stack_va_parts).mean())
print(f"stack valid IC = {base:+.4f}", flush=True)

results = {}
best = (base, 0, 0.0, 0.0, 0.3)
for K in [10, 15, 20, 30]:
    for w0 in [0.9, 1.0]:
        for gamma in [0.9, 1.0]:
            for ahi in [0.7, 0.85]:
                rv = recurse_multi(va, hist_tr[-6:], alpha_hi=ahi)
                parts = list(stack_va_parts)
                for i in range(min(K, len(parts))):
                    w = min(w0 * gamma ** i, 1.0)
                    n = int(va["g"][i])
                    parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rv[i], [n])
                m = float(ics(parts).mean())
                results[f"K{K}_w{w0}_g{gamma}_a{ahi}"] = m
                if m > best[0]:
                    best = (m, K, w0, gamma, ahi)
best_m, K, w0, gamma, ahi = best
print(f"best: K={K} w0={w0} gamma={gamma} alpha_hi={ahi} -> valid {best_m:+.4f} (+{best_m - base:.4f})", flush=True)
top5 = sorted(results.items(), key=lambda kv: -kv[1])[:5]
for k, v in top5:
    print(f"  {k}: {v:+.4f}", flush=True)

# test 手术输出
rec_te_final = recurse_multi(te, hist_va[-6:], alpha_hi=ahi)
parts = list(stack_te_parts)
for i in range(min(K, len(parts))):
    w = min(w0 * gamma ** i, 1.0)
    n = int(te["g"][i])
    parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rec_te_final[i], [n])
final = group_rank(np.concatenate(parts).astype(np.float32), te["g"])
out = vector_to_grid(final, te_time, te_stock, TEST_START, TEST_TIME_POINTS)
mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
mask[te_time - TEST_START, te_stock] = True
contract = validate_prediction(out, mask)
np.save(RESULT / "prediction_1.npy", out)

meta = {
    "experiment": "exp_023f_multilag_surgery",
    "stack_valid_ic": base,
    "recursion": "multi-lag (lag1..lag6) LightGBM + alpha 前置调度（前6截面高置信锚）",
    "grid": results, "best": {"K": K, "w0": w0, "gamma": gamma, "alpha_hi": ahi},
    "blend_valid_ic": best_m, "delta_vs_stack": best_m - base,
    "prev_best": "exp023e online 0.119063 (K=20 线性, 单 lag)",
    "test_modified_sections": f"前 {K} 截面（3161..），其余与 exp021 一致",
    "contract": contract,
    "compliance": "第 t 截面仅用 X(<=t) 与 <t 的给定标签/自有预测；无 t 后数据",
    "elapsed_s": round(time.time() - t0, 1),
}
(RESULT / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("best", "blend_valid_ic", "delta_vs_stack", "contract")}, ensure_ascii=False), flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
