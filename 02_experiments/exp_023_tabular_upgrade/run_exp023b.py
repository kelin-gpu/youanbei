from __future__ import annotations

"""exp_023b：合规递归自举（强因果版）。

机制：y(t-1)+X(t) -> y(t) 近乎确定（oracle IC 0.9785，probe 结论）。test 期无真实
y(t-1)，用自身预测递归 + 收缩混合（a=0.35~0.5 稳定 +0.010，probe 结论）。

合规：第 t 截面输出只由 X(<=t) 与 t-1 及更早的自有预测计算，不使用 t 之后任何数据。
test 递归的 lag 源 = valid 末截面预测（递归跨 valid->test 连续，无断裂）。

产物：valid_pred.npy / test_pred.npy（向量版，与 common 各 split group 对齐）、metrics.json。
"""

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")
RESULT = Path(r"D:\google_dl\book\youanbei\04_results\exp_023b_recursion")
RESULT.mkdir(parents=True, exist_ok=True)

CAP = 768
ROUNDS = 80
PARAMS = dict(objective="huber", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1)
ALPHAS = [0.3, 0.4, 0.5]
LAG_FEED_ALPHA = 0.4  # 递归 lag 喂给下一截面用的档位（固定，避免双重选择）


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
print(f"train matrices {XA.shape}, {time.time()-t0:.0f}s", flush=True)

mA = lgb.train(PARAMS, lgb.Dataset(XA, label=yv), num_boost_round=ROUNDS)
mB = lgb.train(PARAMS, lgb.Dataset(XB, label=yv), num_boost_round=ROUNDS)
del XA, XB
print(f"models done {time.time()-t0:.0f}s", flush=True)

a0, b0 = int(tr["off"][-2]), int(tr["off"][-1])
init_lag = dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist()))


def run_recursion(sp, lag):
    outs = {a: [] for a in ALPHAS}
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        pA = mA.predict(sub)
        pL = mB.predict(np.column_stack([sub, lag_rank(lag, st, st.size)]))
        blended = {al: (1 - al) * pA + al * pL for al in ALPHAS}
        for al, p in blended.items():
            outs[al].append(p)
        lag = dict(zip(st.tolist(), blended[LAG_FEED_ALPHA].tolist()))
    return outs, lag


def section_ics(parts, sp):
    ics = []
    for i, p in enumerate(parts):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        ics.append(np.corrcoef(rankdata(p), rankdata(sp["y"][a:b]))[0, 1])
    return np.asarray(ics)


va_out, lag_after_va = run_recursion(va, init_lag)
# 基线（无 lag 特征）：单独跑一遍 A
va_base = []
lag_b = init_lag
for i in range(len(va["g"])):
    a, b = int(va["off"][i]), int(va["off"][i + 1])
    va_base.append(mA.predict(np.asarray(va["X"][a:b][:, :408], dtype=np.float32)))

base_ics = section_ics(va_base, va)
print(f"baseline A valid IC = {base_ics.mean():+.4f}", flush=True)

metrics = {"alphas": {}, "baseline_valid_ic": float(base_ics.mean())}
for al in ALPHAS:
    ics = section_ics(va_out[al], va)
    metrics["alphas"][str(al)] = {"mean": float(ics.mean()), "pos_rate": float((ics > 0).mean()),
                                  "segments": [float(ics[k * 48:(k + 1) * 48].mean()) for k in range(6)]}
    print(f"alpha={al}: valid IC = {ics.mean():+.4f}", flush=True)
best_alpha = max(ALPHAS, key=lambda a: metrics["alphas"][str(a)]["mean"])

np.save(RESULT / "valid_pred.npy", np.concatenate(va_out[best_alpha]).astype(np.float32))
np.save(RESULT / "valid_pred_base.npy", np.concatenate(va_base).astype(np.float32))

te_out, _ = run_recursion(te, lag_after_va)
te_base = [mA.predict(np.asarray(te["X"][int(te["off"][i]):int(te["off"][i + 1])][:, :408], dtype=np.float32))
           for i in range(len(te["g"]))]
np.save(RESULT / "test_pred.npy", np.concatenate(te_out[best_alpha]).astype(np.float32))
np.save(RESULT / "test_pred_base.npy", np.concatenate(te_base).astype(np.float32))

metrics.update({"best_alpha": best_alpha, "elapsed_s": round(time.time() - t0, 1),
                "compliance": "第 t 截面仅用 X(<=t) 与 t-1 前自有预测；无 t 后数据"})
(RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"done in {time.time()-t0:.0f}s, best_alpha={best_alpha}", flush=True)
