import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")
M023A = Path(r"D:\google_dl\book\youanbei\04_results\exp_023a_future_shift\metrics.json")

CAP = 384
ROUNDS = 60
PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1)

combos = [(d["feat"], d["shift"], abs(d["train_ic"])) for d in json.loads(M023A.read_text())["top20_combos"]][:16]
combos = [(int(f.split("_")[1]), s, w) for f, s, w in combos]


def load(name):
    g = np.load(C / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(C / f"{name}_stock.npy"),
            "y": np.load(C / f"{name}_y.npy"),
            "X": np.load(T / f"{name}_X.npy", mmap_mode="r")}


t0 = time.time()
tr, va = load("train"), load("valid")
n_tr = len(tr["g"])

# 训练输入（一次构建复用）：X(t) cap 抽样
rows, idx_meta = [], []
for i in range(1, n_tr - 20):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    rows.append(np.asarray(tr["X"][a:b][take, :408], dtype=np.float32))
    idx_meta.append((i, tr["s"][a:b][take]))
Xtrain = np.vstack(rows)
print(f"base matrix {Xtrain.shape}, {time.time()-t0:.0f}s", flush=True)

models, signs = [], []
for feat, shift, w in combos:
    tgt = np.full(Xtrain.shape[0], np.nan, dtype=np.float32)
    pos = 0
    for k, (i, st) in enumerate(idx_meta):
        n = st.size
        j = i + shift
        a2, b2 = int(tr["off"][j]), int(tr["off"][j + 1])
        m = dict(zip(tr["s"][a2:b2].tolist(), tr["X"][a2:b2][:, feat].tolist()))
        vals = np.array([m.get(int(s), np.nan) for s in st])
        tgt[pos:pos + n] = vals
        pos += n
    ok = np.isfinite(tgt)
    m_ = lgb.train(PARAMS, lgb.Dataset(Xtrain[ok], label=tgt[ok]), num_boost_round=ROUNDS)
    models.append(m_)
    print(f"  feat {feat}@+{shift}: trainable={ok.mean():.2f}, {time.time()-t0:.0f}s", flush=True)

# valid 评估：X̂(t+shift) 组合 -> y(t)
ics = []
for i in range(len(va["g"])):
    a, b = int(va["off"][i]), int(va["off"][i + 1])
    st = va["s"][a:b]
    sub = np.asarray(va["X"][a:b][:, :408], dtype=np.float32)
    score = np.zeros(st.size)
    for (feat, shift, w), m_ in zip(combos, models):
        xh = m_.predict(sub)
        score += w * rankdata(xh)
    ics.append(np.corrcoef(rankdata(score), rankdata(va["y"][a:b]))[0, 1])
v = np.asarray(ics)
print(f"\ntwo-stage combo valid IC = {v.mean():+.4f}  pos_rate={(v>0).mean():.2f}")
print(f"segments: " + " ".join(f"{v[k*48:(k+1)*48].mean():+.3f}" for k in range(5)), flush=True)
