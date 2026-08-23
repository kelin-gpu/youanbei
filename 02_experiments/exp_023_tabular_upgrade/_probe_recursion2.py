import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")

CAP = 512
ROUNDS = 60
PARAMS = dict(objective="huber", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1)
SS_SPLIT = 1800  # 调度采样: [486,1800) 训 A1 -> 预测 [1800,2918) 作代理 lag -> 训 B2


def load(name):
    g = np.load(C / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(C / f"{name}_stock.npy"),
            "y": np.load(C / f"{name}_y.npy"),
            "X": np.load(T / f"{name}_X.npy", mmap_mode="r")}


def lag_rank_of(src_dict, st, n):
    r = np.full(n, 0.5)
    vals = np.array([src_dict.get(int(s), np.nan) for s in st])
    ok = np.isfinite(vals)
    if ok.sum() > 10:
        r[ok] = (rankdata(vals[ok]) - 1) / (ok.sum() - 1)
    return r.astype(np.float32)


t0 = time.time()
tr, va = load("train"), load("valid")
n_tr = len(tr["g"])
N_STOCK_UNIVERSE = None


def build_train(i0, i1, lag_src_fn):
    rowsA, rowsB, ys = [], [], []
    for i in range(i0, i1):
        a, b = int(tr["off"][i]), int(tr["off"][i + 1])
        take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
        st = tr["s"][a:b][take]
        sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
        lag = lag_src_fn(i, st, st.size)
        rowsA.append(sub)
        rowsB.append(np.column_stack([sub, lag]))
        ys.append(tr["y"][a:b][take])
    return np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)


def true_lag(i, st, n):
    a0, b0 = int(tr["off"][i - 1]), int(tr["off"][i])
    return lag_rank_of(dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist())), st, n)


# ---- Pass 1: A1 on [487,1800) ----
XA1, XB1, y1 = build_train(487, SS_SPLIT, true_lag)
mA1 = lgb.train(PARAMS, lgb.Dataset(XA1, label=y1), num_boost_round=ROUNDS)
mB1 = lgb.train(PARAMS, lgb.Dataset(XB1, label=y1), num_boost_round=ROUNDS)
del XA1, XB1, y1
print(f"pass1 done {time.time()-t0:.0f}s", flush=True)


# ---- A1 对 [1800,2918) 出样本预测 -> 代理 lag 源 ----
proxy = {}
for i in range(SS_SPLIT, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    sub = np.asarray(tr["X"][a:b][:, :408], dtype=np.float32)
    proxy[i] = dict(zip(tr["s"][a:b].tolist(), mA1.predict(sub).tolist()))
print(f"proxy done {time.time()-t0:.0f}s", flush=True)


def proxy_lag(i, st, n):
    return lag_rank_of(proxy.get(i - 1, {}), st, n)


# ---- Pass 2: A 全量 + B2(真lag) + B3(代理lag 调度采样) ----
XA, XBt, yv_ = build_train(487, n_tr, true_lag)
mA = lgb.train(PARAMS, lgb.Dataset(XA, label=yv_), num_boost_round=ROUNDS)
mB2 = lgb.train(PARAMS, lgb.Dataset(XBt, label=yv_), num_boost_round=ROUNDS)
del XA, XBt, yv_
_, XBp, yp = build_train(SS_SPLIT + 1, n_tr, proxy_lag)
mB3 = lgb.train(PARAMS, lgb.Dataset(XBp, label=yp), num_boost_round=ROUNDS)
del XBp, yp
print(f"models done {time.time()-t0:.0f}s", flush=True)

# ---- valid: 递归评估（B2/B3 各自递归 + 收缩混合） ----
a0, b0 = int(tr["off"][-2]), int(tr["off"][-1])
init_lag = dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist()))

configs = {}
for name, model, alpha in [("A", mA, 0.0), ("B2_oracle", mB2, None), ("B2_a0.2", mB2, 0.2),
                            ("B2_a0.35", mB2, 0.35), ("B2_a0.5", mB2, 0.5), ("B2_a0.7", mB2, 0.7),
                            ("B3_rec", mB3, 1.0), ("B3_a0.35", mB3, 0.35), ("B3_a0.5", mB3, 0.5)]:
    lag = init_lag
    ics = []
    for i in range(len(va["g"])):
        a, b = int(va["off"][i]), int(va["off"][i + 1])
        st = va["s"][a:b]
        sub = np.asarray(va["X"][a:b][:, :408], dtype=np.float32)
        pA = mA.predict(sub)
        lr = lag_rank_of(lag, st, st.size)
        if name == "A":
            p = pA
        elif name == "B2_oracle":
            true_src = dict(zip(st.tolist(), va["y"][a:b].tolist()))
            p = model.predict(np.column_stack([sub, lag_rank_of(true_src, st, st.size)]))
        else:
            pL = model.predict(np.column_stack([sub, lr]))
            p = (1 - alpha) * pA + alpha * pL
        ics.append(np.corrcoef(rankdata(p), rankdata(va["y"][a:b]))[0, 1])
        lag = dict(zip(st.tolist(), p.tolist()))
    v = np.asarray(ics)
    configs[name] = v
    print(f"{name:10s}: mean={v.mean():+.4f}  pos={(v>0).mean():.2f}  "
          f"seg: " + " ".join(f"{v[k*50:(k+1)*50].mean():+.3f}" for k in range(5)), flush=True)

print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)
