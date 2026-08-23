import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")

CAP = 512
LAGS = (1, 2, 3, 4, 5, 6)
EVAL_N = 10
SEEDS = (42, 7, 2024)
P = dict(learning_rate=0.05, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, verbosity=-1)


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


def lag_cols(hist_dicts, i, st, mode):
    """mode: rank | rank_raw | rank_raw_diff"""
    cols = []
    raws = []
    for k in LAGS:
        src = hist_dicts[i - k]
        vals = np.array([src.get(int(s), np.nan) for s in st])
        r = np.full(st.size, 0.5, dtype=np.float32)
        ok = np.isfinite(vals)
        if ok.sum() > 10:
            r[ok] = ((rankdata(vals[ok]) - 1) / (ok.sum() - 1)).astype(np.float32)
        cols.append(r)
        raws.append(zscore(vals))
    if mode == "rank":
        return cols
    if mode == "rank_raw":
        return cols + raws
    diffs = [raws[j] - raws[j + 1] for j in range(len(raws) - 1)]
    return cols + raws + diffs


def sec_dicts(sp, i):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    return dict(zip(sp["s"][a:b].tolist(), sp["y"][a:b].tolist()))


def sec_pred_dicts(sp, i, p):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    return dict(zip(sp["s"][a:b].tolist(), p.tolist()))


t0 = time.time()
tr, va = load("train"), load("valid")
n_tr = len(tr["g"])
hist = [sec_dicts(tr, i) for i in range(n_tr)]

# 训练矩阵（三模式）
mats = {"rank": [], "rank_raw": [], "rank_raw_diff": []}
rowsA, ys = [], []
for i in range(7, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    for mode in mats:
        mats[mode].append(np.column_stack([sub] + lag_cols(hist, i, st, mode)))
    rowsA.append(sub)
    ys.append(tr["y"][a:b][take])
yv = np.concatenate(ys)
del rowsA, ys
print(f"matrices {time.time()-t0:.0f}s", flush=True)

MODELS = {}
for mode, parts in mats.items():
    X = np.vstack(parts)
    del parts
    MODELS[mode] = [lgb.train(dict(P, objective="huber", num_leaves=255, min_data_in_leaf=50, seed=s),
                              lgb.Dataset(X, label=yv), num_boost_round=140) for s in SEEDS]
    del X
    print(f"mode {mode} trained, {time.time()-t0:.0f}s", flush=True)


def evaluate(name, mode, seeds_idx=(0,)):
    past = list(hist[-6:])
    ics = []
    for i in range(EVAL_N):
        a, b = int(va["off"][i]), int(va["off"][i + 1])
        st = va["s"][a:b]
        sub = np.asarray(va["X"][a:b][:, :408], dtype=np.float32)
        cols = lag_cols_from_past(past, st, mode)
        Xb = np.column_stack([sub] + cols)
        ps = [MODELS[mode][j].predict(Xb) for j in seeds_idx]
        p = np.mean(ps, axis=0)
        ics.append(np.corrcoef(rankdata(p), rankdata(va["y"][a:b]))[0, 1])
        past.append(sec_pred_dicts(va, i, p))
    v = np.asarray(ics)
    print(f"{name:24s}: first6={v[:6].mean():+.4f}  7-10={v[6:].mean():+.4f}  "
          + " ".join(f"{x:+.3f}" for x in v), flush=True)
    return v


def lag_cols_from_past(past, st, mode):
    cols, raws = [], []
    for k in LAGS:
        src = past[-k] if len(past) >= k else {}
        vals = np.array([src.get(int(s), np.nan) for s in st])
        r = np.full(st.size, 0.5, dtype=np.float32)
        ok = np.isfinite(vals)
        if ok.sum() > 10:
            r[ok] = ((rankdata(vals[ok]) - 1) / (ok.sum() - 1)).astype(np.float32)
        cols.append(r)
        raws.append(zscore(vals))
    if mode == "rank":
        return cols
    if mode == "rank_raw":
        return cols + raws
    return cols + raws + [raws[j] - raws[j + 1] for j in range(len(raws) - 1)]


evaluate("rank deep s42", "rank", (0,))
evaluate("rank deep 3seed", "rank", (0, 1, 2))
evaluate("rank_raw deep s42", "rank_raw", (0,))
evaluate("rank_raw 3seed", "rank_raw", (0, 1, 2))
evaluate("rank_raw_diff 3seed", "rank_raw_diff", (0, 1, 2))
print(f"elapsed {time.time()-t0:.0f}s", flush=True)
