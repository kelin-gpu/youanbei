import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")
CAP, ROUNDS = 768, 80
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
tr, va = load("train"), load("valid")
n_tr = len(tr["g"])
hist_tr = [sec_dicts(tr, i) for i in range(n_tr)]

rowsA, rowsB, ys = [], [], []
for i in range(6, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    rowsA.append(sub)
    rowsB.append(np.column_stack([sub] + [lag_rank(hist_tr[i - k], st, st.size) for k in LAGS]))
    ys.append(tr["y"][a:b][take])
XA, XB, yv = np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)
del rowsA, rowsB
mA = lgb.train(PARAMS, lgb.Dataset(XA, label=yv), num_boost_round=ROUNDS)
mB = lgb.train(PARAMS, lgb.Dataset(XB, label=yv), num_boost_round=ROUNDS)
del XA, XB
print(f"models {time.time()-t0:.0f}s", flush=True)


def recurse(sp, past, alpha_fn):
    outs = []
    past = list(past)
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        n = st.size
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        al = alpha_fn(i)
        p = (1 - al) * mA.predict(sub) + al * mB.predict(
            np.column_stack([sub] + [lag_rank(past[-k] if len(past) >= k else {}, st, n) for k in LAGS]))
        outs.append(p)
        past.append(dict(zip(st.tolist(), p.tolist())))
    return outs


# 首截面分别测纯 mA / 纯 mB / 混合的 IC（诊断锚信息利用率）
first_a = mA.predict(np.asarray(va["X"][:int(va["g"][0])][:, :408], dtype=np.float32))
sub1 = np.asarray(va["X"][:int(va["g"][0])][:, :408], dtype=np.float32)
st1 = va["s"][:int(va["g"][0])]
lag1 = lag_rank(hist_tr[-1], st1, st1.size)
first_b = mB.predict(np.column_stack([sub1] + [lag_rank(hist_tr[-k] if len(hist_tr) >= k else {}, st1, st1.size) for k in LAGS]))
y1 = va["y"][:int(va["g"][0])]
print(f"首截面 IC: 纯mA={np.corrcoef(rankdata(first_a), rankdata(y1))[0,1]:+.4f} "
      f"纯mB={np.corrcoef(rankdata(first_b), rankdata(y1))[0,1]:+.4f}", flush=True)
for al in [0.3, 0.5, 0.7, 0.85, 1.0]:
    p = (1 - al) * first_a + al * first_b
    print(f"  alpha={al}: {np.corrcoef(rankdata(p), rankdata(y1))[0,1]:+.4f}", flush=True)
print(f"diag done {time.time()-t0:.0f}s", flush=True)
