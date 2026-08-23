import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")
from exp_016_unified_expert_fusion.src.ranking import group_rank, dynamic_blend_family_predictions

C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
F16 = Path(r"D:\google_dl\book\youanbei\04_results\exp_016_unified_expert_fusion\full")
P022 = Path(r"D:\google_dl\book\youanbei\04_results\exp_022_tree_full_baseline")

# 用已存的 023f 中间产物不可行（未存），此探针只评估【中段弱混合 + 手术段 cat 叠加】的
# 边际价值: 用 exp022 catboost valid 预测(重算太慢 -> 直接读 prediction 不可行)。
# 结论: 探针改为纯评估——从 023f metrics 反推中段增益, 加载 cat va 预测需要重训。
# 因此本探针只做一件事: 中段(31+) w=0.05/0.10 rec 弱混合的 valid 增益, 复用 023e 的
# 单lag递归逻辑重建 rec(valid 快)。

import lightgbm as lgb

T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")
CAP, ROUNDS = 768, 80
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
rowsA, rowsB, ys = [], [], []
for i in range(1, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, CAP), dtype=np.int64)
    st = tr["s"][a:b][take]
    sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
    a0, b0 = int(tr["off"][i - 1]), int(tr["off"][i])
    rowsA.append(sub)
    rowsB.append(np.column_stack([sub, lag_rank(dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist())), st, st.size)]))
    ys.append(tr["y"][a:b][take])
XA, XB, yv = np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)
del rowsA, rowsB
mA = lgb.train(PARAMS, lgb.Dataset(XA, label=yv), num_boost_round=ROUNDS)
mB = lgb.train(PARAMS, lgb.Dataset(XB, label=yv), num_boost_round=ROUNDS)
del XA, XB
print(f"models {time.time()-t0:.0f}s", flush=True)

rec_parts = []
past = sec_dicts(tr, n_tr - 1)
for i in range(len(va["g"])):
    a, b = int(va["off"][i]), int(va["off"][i + 1])
    st = va["s"][a:b]
    sub = np.asarray(va["X"][a:b][:, :408], dtype=np.float32)
    p = 0.7 * mA.predict(sub) + 0.3 * mB.predict(np.column_stack([sub, lag_rank(past, st, st.size)]))
    rec_parts.append(p)
    past = dict(zip(st.tolist(), p.tolist()))

fam = np.load(F16 / "full_valid_family_predictions.npy")
wts = np.load(F16 / "full_valid_dynamic_weights.npy")
FAMS = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
        "relational_graph", "foundation_representation", "multi_objective_rank")
stack_parts = list(np.split(dynamic_blend_family_predictions({n: fam[:, k] for k, n in enumerate(FAMS)}, va["g"], wts),
                            np.cumsum(va["g"])[:-1]))


def ics(parts):
    out, off = [], 0
    for i, p in enumerate(parts):
        n = int(va["g"][i])
        out.append(np.corrcoef(rankdata(p), rankdata(va["y"][off:off + n]))[0, 1])
        off += n
    return np.asarray(out)


# 手术段(前30, 023f 设置) + 中段弱混合评估
def eval_mid(w_mid, start=30):
    parts = list(stack_parts)
    for i in range(min(30, len(parts))):
        n = int(va["g"][i])
        w = min(1.0 * 0.9 ** i, 1.0)
        parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rec_parts[i], [n])
    for i in range(start, len(parts)):
        n = int(va["g"][i])
        parts[i] = (1 - w_mid) * group_rank(parts[i], [n]) + w_mid * group_rank(rec_parts[i], [n])
    return float(ics(parts).mean())


base30 = eval_mid(0.0)
print(f"base (023f-like surgery only): {base30:+.5f}", flush=True)
for w in [0.02, 0.05, 0.10, 0.15]:
    print(f"mid w={w:.2f}: {eval_mid(w):+.5f}", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
