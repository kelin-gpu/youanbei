import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")

CAP = 512
LAGS = (1, 2, 3, 4, 5, 6)
EVAL_N = 10  # 评估前 10 截面（前 6 真实锚，7-10 部分锚，与 test 同构）
P = dict(learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, verbosity=-1, seed=42)


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
tr, va = load("train"), load("valid")
n_tr = len(tr["g"])
hist = [sec_dicts(tr, i) for i in range(n_tr)]

# ---------- 构建训练矩阵 ----------
rowsA, rowsB, rowsD, rowsE, ys = [], [], [], [], []
for i in range(7, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    # X(t-1)：按 stock 对齐上一截面（全 417 列）
    a0, b0 = int(tr["off"][i - 1]), int(tr["off"][i])
    st_prev = tr["s"][a0:b0]
    pos = {s: j for j, s in enumerate(st_prev.tolist())}
    rows_prev = np.array([pos.get(int(s), -1) for s in st])
    okmask = rows_prev >= 0
    Xprev = np.full((st.size, 419), np.nan, dtype=np.float32)
    Xprev[okmask] = np.asarray(tr["X"][a0:b0], dtype=np.float32)[rows_prev[okmask]]
    lagcols = [lag_rank(hist[i - k], st, st.size) for k in LAGS]
    rowsA.append(sub)
    rowsB.append(np.column_stack([sub] + lagcols))
    rowsD.append(np.column_stack([sub] + lagcols + [Xprev]))
    ys.append(tr["y"][a:b][take])
XA, XB, XD, yv = np.vstack(rowsA), np.vstack(rowsB), np.vstack(rowsD), np.concatenate(ys)
del rowsA, rowsB, rowsD, Xprev
print(f"matrices: A{XA.shape} B{XB.shape} D{XD.shape}, {time.time()-t0:.0f}s", flush=True)

mA = lgb.train(dict(P, objective="huber", num_leaves=63, min_data_in_leaf=100), lgb.Dataset(XA, label=yv), num_boost_round=80)
mB = lgb.train(dict(P, objective="huber", num_leaves=63, min_data_in_leaf=100), lgb.Dataset(XB, label=yv), num_boost_round=80)
mDeep = lgb.train(dict(P, objective="huber", num_leaves=255, min_data_in_leaf=50), lgb.Dataset(XB, label=yv), num_boost_round=140)
mXD = lgb.train(dict(P, objective="huber", num_leaves=255, min_data_in_leaf=50), lgb.Dataset(XD, label=yv), num_boost_round=140)
del XA, XB, XD
print(f"models done, {time.time()-t0:.0f}s", flush=True)

# CatBoost 版（[X, lags]）
import catboost as cb

CAT_COLS = list(range(408, 417))


def cb_pool(X, rel=None):
    numeric = np.ascontiguousarray(X[:, [i for i in range(X.shape[1]) if i not in CAT_COLS]], dtype=np.float32)
    categorical = np.rint(X[:, CAT_COLS]).astype(np.int64).astype(str).astype(object)
    data = cb.FeaturesData(num_feature_data=numeric, cat_feature_data=categorical)
    return cb.Pool(data, label=rel)


def rebuild_B(sp, i, past, st):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
    lagcols = [lag_rank(past[-k] if len(past) >= k else {}, st, st.size) for k in LAGS]
    return np.column_stack([sub] + lagcols)


def rebuild_D(sp, i, past, st):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
    lagcols = [lag_rank(past[-k] if len(past) >= k else {}, st, st.size) for k in LAGS]
    a0, b0 = int(sp["off"][i - 1]), int(sp["off"][i])
    st_prev = sp["s"][a0:b0]
    pos = {s: j for j, s in enumerate(st_prev.tolist())}
    rows_prev = np.array([pos.get(int(s), -1) for s in st])
    okmask = rows_prev >= 0
    Xprev = np.full((st.size, 419), np.nan, dtype=np.float32)
    Xprev[okmask] = np.asarray(sp["X"][a0:b0], dtype=np.float32)[rows_prev[okmask]]
    return np.column_stack([sub] + lagcols + [Xprev])


def eval_model(name, build, model, predict):
    past = list(hist[-6:])
    ics = []
    for i in range(EVAL_N):
        a, b = int(va["off"][i]), int(va["off"][i + 1])
        st = va["s"][a:b]
        Xb = build(va, i, past, st)
        p = predict(model, Xb)
        ics.append(np.corrcoef(rankdata(p), rankdata(va["y"][a:b]))[0, 1])
        past.append(dict(zip(st.tolist(), p.tolist())))
    v = np.asarray(ics)
    print(f"{name:12s}: first6={v[:6].mean():+.4f}  7-10={v[6:].mean():+.4f}  all10={v.mean():+.4f}  "
          + " ".join(f"{x:+.3f}" for x in v), flush=True)
    return v


lgb_pred = lambda m, X: m.predict(X)
results = {}
results["mA(纯X)"] = eval_model("mA", lambda sp, i, past, st: np.asarray(sp["X"][int(sp["off"][i]):int(sp["off"][i+1])][:, :408], dtype=np.float32), mA, lgb_pred)
results["mB(当前)"] = eval_model("mB", rebuild_B, mB, lgb_pred)
results["mDeep"] = eval_model("mDeep", rebuild_B, mDeep, lgb_pred)
results["mXD"] = eval_model("mXD", rebuild_D, mXD, lgb_pred)
print(f"elapsed {time.time()-t0:.0f}s", flush=True)
