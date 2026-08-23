from __future__ import annotations

"""exp_023e：锚点外科手术（targeted anchor surgery）。

核心：test 第 3161 截面的 lag = 真实 y(3160)（给定标签，3160<3161 合规），mB 在该截面
IC 约 0.5-0.9（常规 0.11）。只把 exp021 网格的前 K 个截面按衰减权重混入递归预测，
其余截面保持 exp021 原样。K 与权重曲线在 valid 上同构选择（valid 首截面 2918 的锚点
同样是真实 train 尾 y，结构对称，选择诚实）。
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
RESULT = ROOT / "04_results" / "exp_023e_anchor_surgery"
RESULT.mkdir(parents=True, exist_ok=True)

TEST_START, TEST_TIME_POINTS, STOCK_COUNT = 3161, 442, 5282
FAMILIES = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
            "relational_graph", "foundation_representation", "multi_objective_rank")
CAP = 768
ROUNDS = 80
ALPHA = 0.3
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


t0 = time.time()
tr, va, te = load("train"), load("valid"), load("test")
n_tr = len(tr["g"])

rowsA, rowsB, ys = [], [], []
for i in range(1, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    a0, b0 = int(tr["off"][i - 1]), int(tr["off"][i])
    lag = lag_rank(dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist())), st, st.size)
    rowsA.append(sub)
    rowsB.append(np.column_stack([sub, lag]))
    ys.append(tr["y"][a:b][take])
XA, XB, yv = np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)
del rowsA, rowsB
mA = lgb.train(PARAMS, lgb.Dataset(XA, label=yv), num_boost_round=ROUNDS)
mB = lgb.train(PARAMS, lgb.Dataset(XB, label=yv), num_boost_round=ROUNDS)
del XA, XB
print(f"models done {time.time()-t0:.0f}s", flush=True)


def recurse(sp, lag):
    outs = []
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        p = (1 - ALPHA) * mA.predict(sub) + ALPHA * mB.predict(np.column_stack([sub, lag_rank(lag, st, st.size)]))
        outs.append(p)
        lag = dict(zip(st.tolist(), p.tolist()))
    return outs


a0, b0 = int(tr["off"][-2]), int(tr["off"][-1])
rec_va_parts = recurse(va, dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist())))
vlast = int(va["off"][-1])
n_last = int(va["g"][-1])
rec_te_parts = recurse(te, dict(zip(va["s"][vlast - n_last:vlast].tolist(),
                                    va["y"][vlast - n_last:vlast].tolist())))
print(f"recursion done {time.time()-t0:.0f}s", flush=True)

# 分量：valid 栈 + test 栈（exp021）
fam = np.load(F16 / "full_valid_family_predictions.npy")
wts = np.load(F16 / "full_valid_dynamic_weights.npy")
stack_va_parts = [x for x in np.split(dynamic_blend_family_predictions(
    {n: fam[:, k] for k, n in enumerate(FAMILIES)}, va["g"], wts), np.cumsum(va["g"])[:-1])]
grid021 = np.load(P021)
te_time = np.load(C / "test_time.npy").astype(np.int32)
te_stock = np.load(C / "test_stock.npy").astype(np.int32)
stack_te_vec = grid021[te_time - TEST_START, te_stock]
stack_te_parts = list(np.split(stack_te_vec, np.cumsum(te["g"])[:-1]))

rec_va_ic0 = np.corrcoef(rankdata(rec_va_parts[0]), rankdata(va["y"][:int(va["g"][0])]))[0, 1]
stack_va_ic0 = np.corrcoef(rankdata(stack_va_parts[0]), rankdata(va["y"][:int(va["g"][0])]))[0, 1]
print(f"valid 首截面: rec IC={rec_va_ic0:+.4f}  stack IC={stack_va_ic0:+.4f}", flush=True)


def ics(parts, sp):
    out, off = [], 0
    for i, p in enumerate(parts):
        n = int(sp["g"][i])
        out.append(np.corrcoef(rankdata(p), rankdata(sp["y"][off:off + n]))[0, 1])
        off += n
    return np.asarray(out)


base = ics(stack_va_parts, va)
print(f"stack valid IC = {base.mean():+.4f}", flush=True)

# K 与首截面权重 w0 选择（线性衰减到 0）
results = {}
best = (base.mean(), 0, 0.0)
for K in [1, 2, 3, 5, 8, 12, 20]:
    for w0 in [0.5, 0.7, 0.85, 1.0]:
        parts = list(stack_va_parts)
        for i in range(min(K, len(parts))):
            w = w0 * (1 - i / K)
            n = int(va["g"][i])
            parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rec_va_parts[i], [n])
        m = ics(parts, va).mean()
        results[f"K{K}_w{w0}"] = float(m)
        if m > best[0]:
            best = (float(m), K, w0)
best_m, K, w0 = best
print(f"best surgery: K={K} w0={w0} -> valid {best_m:+.4f} (+{best_m - base.mean():.4f})", flush=True)

# 应用到 test（只改前 K 个截面，其余 exp021 原样）
parts = list(stack_te_parts)
for i in range(min(K, len(parts))):
    w = w0 * (1 - i / K)
    n = int(te["g"][i])
    parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rec_te_parts[i], [n])
final = group_rank(np.concatenate(parts).astype(np.float32), te["g"])
out = vector_to_grid(final, te_time, te_stock, TEST_START, TEST_TIME_POINTS)
mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
mask[te_time - TEST_START, te_stock] = True
contract = validate_prediction(out, mask)
np.save(RESULT / "prediction_1.npy", out)

meta = {
    "experiment": "exp_023e_anchor_surgery",
    "stack_valid_ic": float(base.mean()), "stack_valid_ic_first_section": float(stack_va_ic0),
    "recursion_valid_ic_first_section": float(rec_va_ic0),
    "surgery_grid": results, "best_K": K, "best_w0": w0,
    "blend_valid_ic": float(best_m), "delta_vs_stack": float(best_m - base.mean()),
    "test_modified_sections": f"前 {K} 个截面（3161..），其余与 exp021 逐元素一致",
    "stack_test_source": "exp021 prediction_1.npy (online 0.116568)",
    "contract": contract,
    "compliance": "第 3161 截面仅用 X(3161) 与给定标签 y(3160)；后续截面递归用自有预测；无 t 后数据",
    "elapsed_s": round(time.time() - t0, 1),
}
(RESULT / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("best_K", "best_w0", "blend_valid_ic", "delta_vs_stack", "contract")}, ensure_ascii=False), flush=True)
