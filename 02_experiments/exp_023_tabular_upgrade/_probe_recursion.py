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


def load(name):
    g = np.load(C / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(C / f"{name}_stock.npy"),
            "y": np.load(C / f"{name}_y.npy"),
            "X": np.load(T / f"{name}_X.npy", mmap_mode="r")}


t0 = time.time()
tr, va = load("train"), load("valid")
n_tr = len(tr["g"])

# 训练矩阵（cap 抽样）：A=仅X, B=X+lag1_rank（lag 为真实过去标签，合规）
rowsA, rowsB, ys, gs = [], [], [], []
for i in range(1, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    a0, b0 = int(tr["off"][i - 1]), int(tr["off"][i])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    prev = dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist()))
    lag_raw = np.array([prev.get(int(s), np.nan) for s in st])
    lag_rank = rankdata(lag_raw, nan_policy="omit") if np.isfinite(lag_raw).all() else None
    if lag_rank is None:  # 有缺失股票: 有限值内 rank，缺失填 0.5
        r = np.full(st.size, 0.5)
        ok = np.isfinite(lag_raw)
        r[ok] = (rankdata(lag_raw[ok]) - 1) / max(ok.sum() - 1, 1)
        lag_rank = r
    else:
        lag_rank = (lag_rank - 1) / (st.size - 1)
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    rowsA.append(sub)
    rowsB.append(np.column_stack([sub, lag_rank.astype(np.float32)]))
    ys.append(tr["y"][a:b][take])
    gs.append(st.size)

XA, XB, yv = np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)
del rowsA, rowsB
print(f"train matrix {XA.shape}, {time.time()-t0:.0f}s", flush=True)

dA = lgb.Dataset(XA, label=yv)
dB = lgb.Dataset(XB, label=yv)
mA = lgb.train(PARAMS, dA, num_boost_round=ROUNDS)
mB = lgb.train(PARAMS, dB, num_boost_round=ROUNDS)
del XA, XB, dA, dB
print(f"models trained, {time.time()-t0:.0f}s", flush=True)

# valid 逐截面：A 基线 / B_oracle(真lag上界) / B_rec(递归自举)
icA, icBo, icBr = [], [], []
lag = None  # 递归 lag 源：初始用 train 最后一期真实 y
a0, b0 = int(tr["off"][-2]), int(tr["off"][-1])
lag = dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist()))

for i in range(len(va["g"])):
    a, b = int(va["off"][i]), int(va["off"][i + 1])
    st = va["s"][a:b]
    sub = np.asarray(va["X"][a:b][:, :408], dtype=np.float32)

    def lagcol(src):
        r = np.full(st.size, 0.5)
        vals = np.array([src.get(int(s), np.nan) for s in st])
        ok = np.isfinite(vals)
        r[ok] = (rankdata(vals[ok]) - 1) / max(ok.sum() - 1, 1)
        return r.astype(np.float32), ok.sum()

    pA = mA.predict(sub)
    true_src = dict(zip(st.tolist(), va["y"][a:b].tolist()))
    lc_true, _ = lagcol(true_src)
    pBo = mB.predict(np.column_stack([sub, lc_true]))
    lc_rec, _ = lagcol(lag)
    pBr = mB.predict(np.column_stack([sub, lc_rec]))

    yr = rankdata(va["y"][a:b])
    for p, out in [(pA, icA), (pBo, icBo), (pBr, icBr)]:
        out.append(np.corrcoef(rankdata(p), yr)[0, 1])
    lag = dict(zip(st.tolist(), pBr.tolist()))  # 递归：喂下一步

A, Bo, Br = map(np.asarray, (icA, icBo, icBr))
seg = lambda v, k: f"[{k*50}:{(k+1)*50}]={v[k*50:(k+1)*50].mean():+.4f}"
print(f"\nA  基线(仅X):      mean={A.mean():+.4f}")
print(f"B_oracle(真实lag): mean={Bo.mean():+.4f}  <- lag 代理完美时的上界")
print(f"B_rec(递归自举):   mean={Br.mean():+.4f}  pos_rate={(Br>0).mean():.2f}")
print("B_rec 分段: " + " ".join(seg(Br, k) for k in range(5)), flush=True)
print(f"\nelapsed {time.time()-t0:.0f}s", flush=True)
