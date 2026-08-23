from __future__ import annotations

"""exp_023d：最终合规组合（三分量混合，锚点修复版）。

分量：
  stack  exp021 融合栈（线上 0.116568；valid 用 exp016 full_valid 动态融合输出）
  rec    递归自举锚点修复版（test 第 3161 截面起始 lag = 真实 y(3160)，合规：
         3160 < 3161 <= t；此后递归用自有预测）
  cat    exp022 CatBoost YetiRank（valid 0.0985；valid 用 Phase A 复刻，test 用
         exp022 prediction_2.npy 即 Phase B train+valid 版）

权重：valid 网格选择 + 诚实上限（rec<=0.25 据线上反解 0.09-0.10；cat<=0.30 据树家族
线上比例 1.1-1.15）。无任何 t 后数据。
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
P022 = ROOT / "04_results" / "exp_022_tree_full_baseline" / "prediction_2.npy"
RESULT = ROOT / "04_results" / "exp_023d_final_blend"
RESULT.mkdir(parents=True, exist_ok=True)

TEST_START, TEST_TIME_POINTS, STOCK_COUNT = 3161, 442, 5282
FAMILIES = ("exp015_anchor", "tabular", "dual_axis", "time_frequency",
            "relational_graph", "foundation_representation", "multi_objective_rank")

CAP = 768
ROUNDS = 80
PARAMS = dict(objective="huber", learning_rate=0.05, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbosity=-1)
ALPHA = 0.3  # exp023b 最优档


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


def section_ics(parts, sp):
    out, off = [], 0
    for i, p in enumerate(parts):
        n = int(sp["g"][i])
        out.append(np.corrcoef(rankdata(p), rankdata(sp["y"][off:off + n]))[0, 1])
        off += n
    return np.asarray(out)


t0 = time.time()
tr, va, te = load("train"), load("valid"), load("test")
n_tr = len(tr["g"])

# ---------- 分量 1: 栈 ----------
va_g, va_y = va["g"], va["y"]
fam = np.load(F16 / "full_valid_family_predictions.npy")
wts = np.load(F16 / "full_valid_dynamic_weights.npy")
stack_va = dynamic_blend_family_predictions({n: fam[:, k] for k, n in enumerate(FAMILIES)}, va_g, wts)
grid021 = np.load(P021)
te_time = np.load(C / "test_time.npy").astype(np.int32)
te_stock = np.load(C / "test_stock.npy").astype(np.int32)
stack_te = grid021[te_time - TEST_START, te_stock]
print(f"[1/3] stack ready, {time.time()-t0:.0f}s", flush=True)

# ---------- 分量 2: 递归（锚点修复）----------
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
print(f"[2/3] recursion models done, {time.time()-t0:.0f}s", flush=True)


def recurse(sp, lag):
    outs = []
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        pA = mA.predict(sub)
        pL = mB.predict(np.column_stack([sub, lag_rank(lag, st, st.size)]))
        p = (1 - ALPHA) * pA + ALPHA * pL
        outs.append(p)
        lag = dict(zip(st.tolist(), p.tolist()))
    return outs


a0, b0 = int(tr["off"][-2]), int(tr["off"][-1])
true_tail = dict(zip(tr["s"][a0:b0].tolist(), tr["y"][a0:b0].tolist()))
rec_va_parts = recurse(va, true_tail)          # valid: 与线上同构（起点=真实 lag）
vlast = int(va["off"][-1])
true_va_tail = dict(zip(va["s"][:vlast][-int(va["g"][-1]):].tolist(),
                        va["y"][:vlast][-int(va["g"][-1]):].tolist()))
rec_te_parts = recurse(te, true_va_tail)       # test: 起点改用真实 y(3160)，修复点
rec_va = np.concatenate(rec_va_parts).astype(np.float32)
rec_te = np.concatenate(rec_te_parts).astype(np.float32)
print(f"[2/3] recursion preds done, {time.time()-t0:.0f}s", flush=True)

# ---------- 分量 3: CatBoost Phase A（valid 诚实预测）----------
import catboost as cb

CAT_COLS = list(range(408, 417))
rows, rels = [], []
for i in range(0, n_tr):
    a, b = int(tr["off"][i]), int(tr["off"][i + 1])
    take = np.linspace(0, b - a - 1, min(b - a, 1024), dtype=np.int64)
    rows.append(np.asarray(tr["X"][a:b][take], dtype=np.float32))
    rels.append(np.rint(tr["y"][a:b][take]).astype(np.int64))
Xc = np.vstack(rows)
rc = np.concatenate(rels)
gc = np.full(Xc.shape[0], 0, dtype=np.int64)
pos = 0
for i, r in enumerate(rels):
    gc[pos:pos + r.size] = i
    pos += r.size
del rows, rels
print(f"[3/3] catboost matrix {Xc.shape}, {time.time()-t0:.0f}s", flush=True)


def cb_pool(X, rel=None, groups=None):
    numeric = np.ascontiguousarray(X[:, [i for i in range(X.shape[1]) if i not in CAT_COLS]], dtype=np.float32)
    categorical = np.rint(X[:, CAT_COLS]).astype(np.int64).astype(str).astype(object)
    data = cb.FeaturesData(num_feature_data=numeric, cat_feature_data=categorical)
    gid = None if groups is None else groups
    return cb.Pool(data, label=rel, group_id=gid)


mcat = cb.train(cb_pool(Xc, rc, gc), {"loss_function": "YetiRank", "learning_rate": 0.05,
                                      "depth": 6, "verbose": False, "allow_writing_files": False,
                                      "random_seed": 42, "iterations": 55})
del Xc
cat_va_parts = []
for i in range(len(va["g"])):
    a, b = int(va["off"][i]), int(va["off"][i + 1])
    cat_va_parts.append(mcat.predict(cb_pool(np.asarray(va["X"][a:b], dtype=np.float32))).astype(np.float32))
cat_va = np.concatenate(cat_va_parts)
grid022 = np.load(P022)
cat_te = grid022[te_time - TEST_START, te_stock]
print(f"[3/3] catboost done, {time.time()-t0:.0f}s", flush=True)

# ---------- 混合权重选择 ----------
stack_ics = section_ics(np.split(stack_va, np.cumsum(va["g"])[:-1]), va)
rec_ics = section_ics(rec_va_parts, va)
cat_ics = section_ics(cat_va_parts, va)
rs, rr, rc_ = group_rank(stack_va, va_g), group_rank(rec_va, va_g), group_rank(cat_va, va_g)
print(f"component valid IC: stack={stack_ics.mean():+.4f} rec={rec_ics.mean():+.4f} cat={cat_ics.mean():+.4f}", flush=True)
print(f"corr: s-r={np.corrcoef(rs, rr)[0,1]:.3f} s-c={np.corrcoef(rs, rc_)[0,1]:.3f} r-c={np.corrcoef(rr, rc_)[0,1]:.3f}", flush=True)


def blend_valid(w_s, w_r, w_c):
    v = w_s * rs + w_r * rr + w_c * rc_
    return section_ics(np.split(v, np.cumsum(va["g"])[:-1]), va).mean()


grid_results = {}
best = (stack_ics.mean(), 1.0, 0.0, 0.0)
for w_r in [0.0, 0.10, 0.15, 0.20, 0.25]:
    for w_c in [0.0, 0.10, 0.20, 0.30]:
        w_s = 1.0 - w_r - w_c
        if w_s < 0.40:
            continue
        m = blend_valid(w_s, w_r, w_c)
        grid_results[f"s{w_s:.2f}_r{w_r:.2f}_c{w_c:.2f}"] = float(m)
        if m > best[0]:
            best = (float(m), w_s, w_r, w_c)
best_m, w_s, w_r, w_c = best
print(f"best honest-capped weights: stack={w_s:.2f} rec={w_r:.2f} cat={w_c:.2f} -> valid {best_m:+.4f} "
      f"(stack alone {stack_ics.mean():+.4f}, +{best_m - stack_ics.mean():.4f})", flush=True)

# ---------- test 输出 ----------
te_g = te["g"]
blend_te = w_s * group_rank(stack_te, te_g) + w_r * group_rank(rec_te, te_g) + w_c * group_rank(cat_te, te_g)
final = group_rank(blend_te, te_g)
out = vector_to_grid(final, te_time, te_stock, TEST_START, TEST_TIME_POINTS)
mask = np.zeros((TEST_TIME_POINTS, STOCK_COUNT), dtype=bool)
mask[te_time - TEST_START, te_stock] = True
contract = validate_prediction(out, mask)
np.save(RESULT / "prediction_1.npy", out)

meta = {
    "experiment": "exp_023d_final_blend",
    "components": {
        "stack": {"valid_ic": float(stack_ics.mean()), "online": 0.116568},
        "recursion_fixed_anchor": {"valid_ic": float(rec_ics.mean()), "online_est": "0.095-0.10 (back-solved + anchor fix)"},
        "catboost_yetirank": {"valid_ic": float(cat_ics.mean()), "online_est": "0.108-0.113 (tree-family ratio)"},
    },
    "weights": {"stack": float(w_s), "rec": float(w_r), "cat": float(w_c)},
    "blend_valid_ic": float(best_m), "delta_vs_stack": float(best_m - stack_ics.mean()),
    "grid": grid_results,
    "anchor_fix": "test 递归起点 lag = 真实 y(3160)（valid 标签为给定数据，3160<t 合规）",
    "contract": contract,
    "compliance": "所有分量第 t 截面仅用 X(<=t) 与 <=t-1 的给定标签/自有预测；无 t 后数据",
    "elapsed_s": round(time.time() - t0, 1),
}
(RESULT / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: meta[k] for k in ("weights", "blend_valid_ic", "delta_vs_stack", "contract")}, ensure_ascii=False), flush=True)
print(f"done in {time.time()-t0:.0f}s", flush=True)
