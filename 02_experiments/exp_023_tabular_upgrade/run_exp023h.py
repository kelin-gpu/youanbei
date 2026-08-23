from __future__ import annotations

"""exp_023h：锚点手术终极版（mDeep + rank_raw lag + 3种子 + 扩展网格）。

栈 = exp021 test 网格（线上 0.116568，已验证不可再动）。
手术 = 深度 LightGBM（num_leaves 255, 140轮）× 3 种子平均，lag 输入 rank+原始 z-score，
       alpha_hi 网格扩展到 1.0，K/γ 网格加宽。
验收：valid 手术 IC >= 0.0990 才建议提交（023f 为 0.0987，线上 0.119533）。
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")
from exp_016_unified_expert_fusion.src.ranking import group_rank
from exp_016_unified_expert_fusion.src.prediction_contract import vector_to_grid, validate_prediction

ROOT = Path(r"D:\google_dl\book\youanbei")
C = ROOT / "03_cache" / "processed_data_v1" / "common"
T = ROOT / "03_cache" / "processed_data_v1" / "tree"
F16 = ROOT / "04_results" / "exp_016_unified_expert_fusion" / "full"
P021 = ROOT / "04_results" / "exp_021_retrain_head_router" / "prediction_1.npy"
RESULT = ROOT / "04_results" / "exp_023h_ultimate_surgery"
RESULT.mkdir(parents=True, exist_ok=True)

TEST_START, TEST_TIME_POINTS, STOCK_COUNT = 3161, 442, 5282
CAP = 768
ROUNDS = 140
LAGS = (1, 2, 3, 4, 5, 6)
SEEDS = (42, 7, 2024)
ALPHA_LO = 0.3
P_DEEP = dict(learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1, objective="huber",
              num_leaves=255, min_data_in_leaf=50)
GRID = {"alpha_hi": [0.7, 0.85, 1.0], "K": [6, 10, 15, 20, 30, 40], "gamma": [0.85, 0.9, 0.95]}


def load(name):
    g = np.load(C / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(C / f"{name}_stock.npy"),
            "y": np.load(C / f"{name}_y.npy") if name != "test" else None,
            "X": np.load(T / f"{name}_X.npy", mmap_mode="r")}


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    out = np.zeros(v.size, dtype=np.float32)
    if ok.sum() > 10:
        mu, sd = v[ok].mean(), v[ok].std()
        out[ok] = ((v[ok] - mu) / (sd if sd > 1e-9 else 1.0)).astype(np.float32)
    return out


def lag_cols_from_dicts(src_dicts, st):
    """rank + 原始 z-score 两种 lag 编码。src_dicts 按 LAGS 顺序给出 lag1..lag6 的 dict。"""
    cols, raws = [], []
    for src in src_dicts:
        vals = np.array([src.get(int(s), np.nan) for s in st])
        r = np.full(st.size, 0.5, dtype=np.float32)
        ok = np.isfinite(vals)
        if ok.sum() > 10:
            r[ok] = ((rankdata(vals[ok]) - 1) / (ok.sum() - 1)).astype(np.float32)
        cols.append(r)
        raws.append(zscore(vals))
    return cols + raws


def sec_dicts(sp, i):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    return dict(zip(sp["s"][a:b].tolist(), sp["y"][a:b].tolist()))


t0 = time.time()
tr, va, te = load("train"), load("valid"), load("test")
n_tr = len(tr["g"])
hist = [sec_dicts(tr, i) for i in range(n_tr)]

# ---- 训练矩阵（rank_raw lag）----
rows, ys = [], []
for i in range(7, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    lagd = [hist[i - k] for k in LAGS]
    rows.append(np.column_stack([sub] + lag_cols_from_dicts(lagd, st)))
    ys.append(tr["y"][a:b][take])
XB, yv = np.vstack(rows), np.concatenate(ys)
del rows
print(f"train matrix {XB.shape}, {time.time()-t0:.0f}s", flush=True)

mBs = [lgb.train(dict(P_DEEP, seed=s), lgb.Dataset(XB, label=yv), num_boost_round=ROUNDS) for s in SEEDS]
del XB
CKPT = ROOT / "03_cache" / "exp_023h_ultimate_surgery"
CKPT.mkdir(parents=True, exist_ok=True)
for j, m in enumerate(mBs):
    m.save_model(str(CKPT / f"mB_{SEEDS[j]}.txt"))
print(f"3-seed deep models done, {time.time()-t0:.0f}s", flush=True)


def predict_mB(sub, lagd, st):
    Xb = np.column_stack([sub] + lag_cols_from_dicts(lagd, st))
    return np.mean([m.predict(Xb) for m in mBs], axis=0)


def recurse(sp, past, alpha_hi):
    """past: 先前截面的 dict 列表（末尾为 t-1）。lag 链用纯 mB 输出。"""
    outs = []
    past = list(past)
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        lagd = [past[-k] if len(past) >= k else {} for k in LAGS]
        p = predict_mB(sub, lagd, st)
        outs.append(p)
        past.append(dict(zip(st.tolist(), p.tolist())))
    return outs


# ---- valid：栈向量 + 手术网格 ----
va_g = np.load(C / "valid_group_sizes.npy")
va_y = va["y"]
fam = np.load(F16 / "full_valid_family_predictions.npy")
wts = np.load(F16 / "full_valid_dynamic_weights.npy")
FAMS = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
        "relational_graph", "foundation_representation", "multi_objective_rank")
from exp_016_unified_expert_fusion.src.ranking import dynamic_blend_family_predictions
stack_va = dynamic_blend_family_predictions({n: fam[:, k] for k, n in enumerate(FAMS)}, va_g, wts)
stack_parts = list(np.split(stack_va, np.cumsum(va_g)[:-1]))


def ics_of(parts):
    out, off = [], 0
    for i, p in enumerate(parts):
        n = int(va_g[i])
        out.append(np.corrcoef(rankdata(p), rankdata(va_y[off:off + n]))[0, 1])
        off += n
    return np.asarray(out)


base = float(ics_of(stack_parts).mean())
print(f"stack valid IC = {base:+.6f}", flush=True)

results = {}
best = (base, 0, 0.0, 0.0)
for ahi in GRID["alpha_hi"]:
    rv = recurse(va, hist[-6:], ahi)
    for K in GRID["K"]:
        for gamma in GRID["gamma"]:
            parts = list(stack_parts)
            for i in range(min(K, len(parts))):
                w = min(1.0 * gamma ** i, 1.0)
                n = int(va_g[i])
                if i < 6:
                    # 前 6 截面：栈 rank 与 mB rank 按 alpha_hi 混合后作为替换值
                    blend_sec = (1 - ahi) * group_rank(parts[i], [n]) + ahi * group_rank(rv[i], [n])
                else:
                    blend_sec = (1 - ALPHA_LO) * group_rank(parts[i], [n]) + ALPHA_LO * group_rank(rv[i], [n])
                parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * blend_sec
            m = float(ics_of(parts).mean())
            results[f"a{ahi}_K{K}_g{gamma}"] = m
            if m > best[0]:
                best = (m, K, gamma, ahi)
best_m, K, gamma, ahi = best
print(f"best: alpha_hi={ahi} K={K} gamma={gamma} -> valid {best_m:+.6f} (+{best_m-base:.6f})", flush=True)
for k, v in sorted(results.items(), key=lambda kv: -kv[1])[:6]:
    print(f"  {k}: {v:+.6f}", flush=True)

# ---- test ----
te_time = np.load(C / "test_time.npy").astype(np.int32)
te_stock = np.load(C / "test_stock.npy").astype(np.int32)
grid021 = np.load(P021)
stack_te = grid021[te_time - TEST_START, te_stock]
te_parts = list(np.split(stack_te, np.cumsum(te["g"])[:-1]))

hist_va = [sec_dicts(va, i) for i in range(len(va["g"]))]
rv_te = recurse(te, hist_va[-6:], ahi)
for i in range(min(K, len(te_parts))):
    w = min(1.0 * gamma ** i, 1.0)
    n = int(te["g"][i])
    if i < 6:
        blend_sec = (1 - ahi) * group_rank(te_parts[i], [n]) + ahi * group_rank(rv_te[i], [n])
    else:
        blend_sec = (1 - ALPHA_LO) * group_rank(te_parts[i], [n]) + ALPHA_LO * group_rank(rv_te[i], [n])
    te_parts[i] = (1 - w) * group_rank(te_parts[i], [n]) + w * blend_sec
final = group_rank(np.concatenate(te_parts).astype(np.float32), te["g"])
out = vector_to_grid(final, te_time, te_stock, TEST_START, TEST_TIME_POINTS)
mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
mask[te_time - TEST_START, te_stock] = True
contract = validate_prediction(out, mask)
np.save(RESULT / "prediction_1.npy", out)

meta = {
    "experiment": "exp_023h_ultimate_surgery",
    "stack_valid_ic": base,
    "stack_test_source": "exp021 prediction_1.npy (online 0.116568)",
    "recursion": "mDeep num_leaves=255 140轮 x 3种子平均, lag=rank+raw z-score",
    "grid": results,
    "best": {"alpha_hi": ahi, "K": K, "gamma": gamma},
    "surgery_valid_ic": best_m,
    "delta_vs_stack": best_m - base,
    "delta_vs_023f_valid": best_m - 0.098683,
    "contract": contract,
    "compliance": "第 t 截面仅用 X(<=t) 与 <t 给定标签/自有预测；无 t 后数据",
    "elapsed_s": round(time.time() - t0, 1),
}
(RESULT / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("best", "surgery_valid_ic", "delta_vs_023f_valid", "contract")}, ensure_ascii=False), flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
