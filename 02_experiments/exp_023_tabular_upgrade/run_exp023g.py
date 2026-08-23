from __future__ import annotations

"""exp_023g：CatBoost tabular 专家替换 + 全栈重训 + 锚点手术（冲击 0.12 收官）。

三段增益组合（区段不相交，近似可加）：
  1. tabular 家族升级：exp016 的 4×16 轮弱集成 -> exp022 配方 CatBoost YetiRank
     （lr 0.05, depth 6, 60 轮 = Phase A best 55 × 1.1，无早停不看 valid，防泄漏）。
     standalone full-valid IC 0.0985（exp022 实测）vs 旧 tabular ~0.093。
  2. head/router 一致重训（exp021 方法论：OOF×3 重算 states，端到端重训 head+router）。
     exp016(0.116132) -> exp020(0.116252) -> exp021(0.116568) 已验证该管线迁移稳定。
  3. 锚点手术（exp023f 配方：多 lag 递归 + alpha 前置调度，K/w0/gamma 在新栈
     valid 向量上重选；test 前 K 截面替换，其余保持新栈）。

合规：所有训练/推理第 t 截面仅用 X(<=t) 与 <t 的给定标签；手术段锚点为给定历史
标签 y(3160) 等；无任何 t 后数据。CatBoost 轮数 60 来自 exp022 Phase A（valid
早停 best=55），Phase B 固定轮数协议与 exp022 一致。

阶段：
  stage=fit    OOF×3 + official_valid CatBoost 重训，head+router 重训，
               capped valid 评估 + full-valid 重融合参照 + 手术参数选择
  stage=submit final CatBoost 重训 -> Test 新栈网格 -> 手术叠加 -> prediction_1.npy
  stage=all    两者都做

训练保护：DSCR_EXP016_MODE=full 且 DSCR_EXP016_ALLOW_TRAINING=YES。
输出只写 exp_023g 自己目录，不覆盖 exp016/exp021/final_submission。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")

from exp_016_unified_expert_fusion.config import (BASE_WEIGHTS, CACHE_DIR, FAMILIES, MIN_WEIGHTS,
                                                  OOF_FOLDS, RESULT_DIR, RunConfig, STOCK_COUNT,
                                                  TEST_START, TEST_STOP, VALID_START, VALID_STOP,
                                                  require_training)
from exp_016_unified_expert_fusion.src.data_context import DataContext
from exp_016_unified_expert_fusion.src.full_pipeline import (_collect_head_inputs, _combined_batch_factory,
                                                             _combined_supervised_arrays, _head_outputs_by_group,
                                                             _period_sample, _tabular_arrays, _torch_batches)
from exp_016_unified_expert_fusion.src.multi_objective_head import MultiObjectiveRankHead
from exp_016_unified_expert_fusion.src.oof_anchor import (load_exp015_anchor, load_exp015_test_anchor,
                                                          predict_exp015_anchor)
from exp_016_unified_expert_fusion.src.prediction_contract import validate_prediction, vector_to_grid
from exp_016_unified_expert_fusion.src.ranking import dynamic_blend_family_predictions, rank_ic, group_rank
from exp_016_unified_expert_fusion.src.state_router import StateRouter
from exp_016_unified_expert_fusion.src.tabular_experts import _catboost_pool, validate_relevance
from exp_016_unified_expert_fusion.src.time_frequency import fit_period_library
from exp_016_unified_expert_fusion.src.training import restore_torch_checkpoint, train_head, train_router

EXP016_FULL = RESULT_DIR / "full"
EXP016_CACHE = CACHE_DIR
RESULT = Path(r"D:\google_dl\book\youanbei\04_results\exp_023g_catboost_stack_surgery")
CKPT = Path(r"D:\google_dl\book\youanbei\03_cache\exp_023g_catboost_stack")
C = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\common")
T = Path(r"D:\google_dl\book\youanbei\03_cache\processed_data_v1\tree")

CAT_PARAMS = {"loss_function": "YetiRank", "learning_rate": 0.05, "depth": 6,
              "iterations": 60, "verbose": False, "allow_writing_files": False, "random_seed": 42}

SURGERY_GRID = {"K": [6, 10, 15, 20, 30], "w0": [0.9, 1.0], "gamma": [0.9, 1.0], "alpha_hi": [0.7, 0.85]}
RECURSION = {"CAP": 768, "ROUNDS": 80, "LAGS": (1, 2, 3, 4, 5, 6), "alpha_lo": 0.3,
             "params": dict(objective="huber", learning_rate=0.05, num_leaves=63,
                            min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                            bagging_freq=1, verbosity=-1, seed=42)}


def per_group_ic(pred, target, groups):
    scores, offset = [], 0
    for size in groups:
        size = int(size)
        scores.append(rank_ic(pred[offset:offset + size], target[offset:offset + size]))
        offset += size
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


# ---------------- CatBoost tabular（exp022 配方，接口与 tabular_experts 对齐） ----------------

def train_cat_tabular(config, X, target, relevance, groups):
    require_training(config, "CatBoost tabular 专家训练")
    import catboost as cb
    X = np.asarray(X, np.float32)
    relevance = np.asarray(relevance, np.int32)
    groups = np.asarray(groups, np.int32)
    if X.shape[0] != np.asarray(target).size or int(groups.sum()) != X.shape[0]:
        raise ValueError("CatBoost 训练数组行契约不一致。")
    validate_relevance(relevance, groups)
    return {"catboost": cb.train(_catboost_pool(X, relevance, groups), dict(CAT_PARAMS))}


def predict_cat_tabular(models, X, groups):
    X = np.asarray(X, np.float32)
    pred = np.asarray(models["catboost"].predict(_catboost_pool(X, None, None)), np.float32)
    return group_rank(pred, groups).astype(np.float32), {"catboost": pred}


# ---------------- 递归自举（exp023f 同配方） ----------------

def _load_split(name):
    g = np.load(C / f"{name}_group_sizes.npy")
    return {"g": g, "off": np.concatenate([[0], np.cumsum(g)]),
            "s": np.load(C / f"{name}_stock.npy"),
            "y": np.load(C / f"{name}_y.npy") if name != "test" else None,
            "X": np.load(T / f"{name}_X.npy", mmap_mode="r")}


def _lag_rank(src, st, n):
    r = np.full(n, 0.5, dtype=np.float32)
    vals = np.array([src.get(int(s), np.nan) for s in st])
    ok = np.isfinite(vals)
    if ok.sum() > 10:
        r[ok] = ((rankdata(vals[ok]) - 1) / (ok.sum() - 1)).astype(np.float32)
    return r


def _sec_dicts(sp, i):
    a, b = int(sp["off"][i]), int(sp["off"][i + 1])
    return dict(zip(sp["s"][a:b].tolist(), sp["y"][a:b].tolist()))


def train_recursion_models(tr):
    import lightgbm as lgb
    n_tr = len(tr["g"])
    rowsA, rowsB, ys = [], [], []
    hist = [_sec_dicts(tr, i) for i in range(n_tr)]
    lags = RECURSION["LAGS"]
    cap = RECURSION["CAP"]
    for i in range(6, n_tr):
        a, b = int(tr["off"][i]), int(tr["off"][i + 1])
        take = np.linspace(0, b - a - 1, min(b - a, cap), dtype=np.int64)
        st = tr["s"][a:b][take]
        sub = np.asarray(tr["X"][a:b][take, :408], dtype=np.float32)
        lagcols = [_lag_rank(hist[i - k], st, st.size) for k in lags]
        rowsA.append(sub)
        rowsB.append(np.column_stack([sub] + lagcols))
        ys.append(tr["y"][a:b][take])
    XA, XB, yv = np.vstack(rowsA), np.vstack(rowsB), np.concatenate(ys)
    del rowsA, rowsB
    mA = lgb.train(RECURSION["params"], lgb.Dataset(XA, label=yv), num_boost_round=RECURSION["ROUNDS"])
    mB = lgb.train(RECURSION["params"], lgb.Dataset(XB, label=yv), num_boost_round=RECURSION["ROUNDS"])
    del XA, XB
    return mA, mB


def recurse_multi(mA, mB, sp, past, alpha_hi):
    outs = []
    past = list(past)
    lags = RECURSION["LAGS"]
    alpha_lo = RECURSION["alpha_lo"]
    for i in range(len(sp["g"])):
        a, b = int(sp["off"][i]), int(sp["off"][i + 1])
        st = sp["s"][a:b]
        n = st.size
        sub = np.asarray(sp["X"][a:b][:, :408], dtype=np.float32)
        pA = mA.predict(sub)
        lagcols = [_lag_rank(past[-k] if len(past) >= k else {}, st, n) for k in lags]
        pL = mB.predict(np.column_stack([sub] + lagcols))
        al = alpha_hi if i < 6 else alpha_lo
        p = (1 - al) * pA + al * pL
        outs.append(p)
        past.append(dict(zip(st.tolist(), p.tolist())))
    return outs


# ---------------- fit 阶段 ----------------

def fit_stage(config: RunConfig, ctx: DataContext) -> dict:
    t0 = time.time()
    cap = config.stock_cap
    family_matrix = np.load(EXP016_CACHE / "oof_predictions" / "family_matrix.npy").astype(np.float32)

    experts_parts, target_parts, state_parts, group_parts = [], [], [], []
    offset = 0
    for fold_name, ts, te, ps, pe in OOF_FOLDS:
        print(f"[exp023g] fold {fold_name}: retrain CatBoost tabular", flush=True)
        train_X, train_y, train_rel, train_groups, _ = _tabular_arrays(ctx, "train", ts, te, cap)
        models = train_cat_tabular(config, train_X, train_y, train_rel, train_groups)
        anchor_model = load_exp015_anchor(EXP016_CACHE / "checkpoints" / fold_name / "anchor.txt")
        train_anchor = predict_exp015_anchor(anchor_model, train_X)
        pred_X, pred_y, pred_rel, pred_groups, _ = _tabular_arrays(ctx, "train", ps, pe, cap)
        pred_anchor = predict_exp015_anchor(anchor_model, pred_X)
        tabular_pred, _ = predict_cat_tabular(models, pred_X, pred_groups)
        del train_X, pred_X, models

        raw_factory = lambda ts2=ts, te2=te, av=train_anchor: _torch_batches(ctx, "train", ts2, te2, cap, av)
        history, mask = _period_sample(raw_factory)
        periods = fit_period_library(history, mask)

        rows = tabular_pred.size
        new_six = family_matrix[offset:offset + rows].copy()
        new_six[:, 1] = tabular_pred
        offset += rows
        family6 = {FAMILIES[i]: new_six[:, i] for i in range(6)}

        pred_factory = lambda ps2=ps, pe2=pe, av=pred_anchor, pl=periods: _torch_batches(ctx, "train", ps2, pe2, cap, av, pl)
        experts, target, states, groups = _collect_head_inputs(pred_factory, family6)
        experts_parts.append(experts); target_parts.append(target); state_parts.append(states); group_parts.append(groups)

    experts = torch.cat(experts_parts)
    target = torch.cat(target_parts)
    states = torch.cat(state_parts)
    groups = torch.cat(group_parts)
    print(f"[exp023g] OOF rows={experts.shape[0]}, groups={groups.numel()}, elapsed={time.time()-t0:.1f}s", flush=True)

    CKPT.mkdir(parents=True, exist_ok=True)
    head = MultiObjectiveRankHead(expert_count=6, state_dim=8)
    head_batches, offset = [], 0
    for gi, g in enumerate(groups.tolist()):
        g = int(g)
        state_block = states[gi:gi + 1].repeat(g, 1)
        head_batches.append((experts[offset:offset + g], state_block, target[offset:offset + g], groups[gi:gi + 1]))
        offset += g
    train_head(config, head, head_batches, CKPT / "multi_objective_head.pt")
    head.to("cpu")

    output = _head_outputs_by_group(head, experts, states, groups)
    family7 = {FAMILIES[i]: experts[:, i].numpy() for i in range(6)}
    family7["multi_objective_rank"] = output["score"].numpy().astype(np.float32)
    ranked = np.stack([group_rank(family7[name], groups.numpy()) for name in FAMILIES], axis=1)
    desired = []
    offset = 0
    for g in groups.tolist():
        g = int(g)
        block_y, block_p = target[offset:offset + g], torch.from_numpy(ranked[offset:offset + g])
        quality = []
        for col in range(block_p.shape[1]):
            left = block_p[:, col] - block_p[:, col].mean()
            right = block_y - block_y.mean()
            quality.append(torch.clamp((left * right).mean() / (left.std(unbiased=False).clamp_min(1e-6) * right.std(unbiased=False).clamp_min(1e-6)), -1, 1))
        score = torch.softmax(torch.stack(quality), dim=0)
        minimum = torch.tensor([MIN_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
        desired.append(minimum + (1.0 - minimum.sum()) * score)
        offset += g
    base = torch.tensor([BASE_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    minimum = torch.tensor([MIN_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    router = StateRouter(8, len(FAMILIES), base, minimum)
    train_router(config, router, states, torch.stack(desired), CKPT / "state_router.pt")
    router.to("cpu")

    # capped valid 评估（与 exp016 0.090487 / exp021 0.090972 同口径）
    valid_anchor_model = load_exp015_anchor(EXP016_CACHE / "checkpoints" / "official_valid" / "anchor.txt")
    valid_X, valid_y, _, valid_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, cap)
    valid_anchor = predict_exp015_anchor(valid_anchor_model, valid_X)
    print("[exp023g] retrain official_valid CatBoost", flush=True)
    models_val = train_cat_tabular(config, *(_tabular_arrays(ctx, "train", 486, VALID_START, cap)[:4]))
    tabular_valid, _ = predict_cat_tabular(models_val, valid_X, valid_groups)

    valid_six = np.load(EXP016_CACHE / "oof_predictions" / "official_valid_first_six.npy").astype(np.float32)
    assert valid_six.shape[0] == valid_anchor.size, "valid first-six 与 capped valid 行数不一致"
    new_valid_six = valid_six.copy()
    new_valid_six[:, 1] = tabular_valid
    valid_family6 = {FAMILIES[i]: new_valid_six[:, i] for i in range(6)}

    train_X, _, _, _, _ = _tabular_arrays(ctx, "train", 486, VALID_START, cap)
    train_anchor_ov = predict_exp015_anchor(valid_anchor_model, train_X)
    del train_X
    raw_train = lambda: _torch_batches(ctx, "train", 486, VALID_START, cap, train_anchor_ov)
    history, mask = _period_sample(raw_train)
    periods_valid = fit_period_library(history, mask)
    valid_factory = lambda: _torch_batches(ctx, "valid", VALID_START, VALID_STOP, cap, valid_anchor, periods_valid)
    experts_v, _, states_v, groups_v = _collect_head_inputs(valid_factory, valid_family6)

    output_v = _head_outputs_by_group(head, experts_v, states_v, groups_v)
    family7_v = {FAMILIES[i]: valid_family6[FAMILIES[i]] for i in range(6)}
    family7_v["multi_objective_rank"] = output_v["score"].numpy().astype(np.float32)
    router.to("cpu").eval()
    with torch.no_grad():
        confidence_v = torch.stack([part.mean() for part in torch.split(output_v["confidence"], groups_v.tolist())])
        weights_v = router(states_v, confidence_v)
    blended_v = dynamic_blend_family_predictions(family7_v, groups_v.numpy(), weights_v.numpy())
    capped_ic = per_group_ic(blended_v, valid_y, valid_groups)
    del valid_X

    # full valid：CatBoost standalone + 重融合参照 + 手术参数选择
    full_X, full_y, _, full_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, STOCK_COUNT)
    tabular_full, _ = predict_cat_tabular(models_val, full_X, full_groups)
    del full_X, models_val
    cat_standalone_ic = per_group_ic(tabular_full, full_y, full_groups)

    full_family = np.load(EXP016_FULL / "full_valid_family_predictions.npy").copy()
    full_w = np.load(EXP016_FULL / "full_valid_dynamic_weights.npy")
    full_family[:, 1] = tabular_full
    valid_family = {name: full_family[:, i] for i, name in enumerate(FAMILIES)}
    stack_va = dynamic_blend_family_predictions(valid_family, full_groups, full_w)
    reblend_ic = per_group_ic(stack_va, full_y, full_groups)
    print(f"[exp023g] catboost standalone full-valid IC = {cat_standalone_ic:.6f}", flush=True)
    print(f"[exp023g] re-blended full-valid IC = {reblend_ic:.6f} (exp016 router 0.091389)", flush=True)

    # 递归模型 + 手术参数选择（对新栈 valid 向量）
    tr = _load_split("train")
    va = _load_split("valid")
    mA, mB = train_recursion_models(tr)
    mA.save_model(str(CKPT / "recursion_mA.txt"))
    mB.save_model(str(CKPT / "recursion_mB.txt"))

    stack_parts = list(np.split(stack_va, np.cumsum(full_groups)[:-1]))

    def ics_of(parts):
        out, off = [], 0
        for i, p in enumerate(parts):
            n = int(full_groups[i])
            out.append(np.corrcoef(rankdata(p), rankdata(full_y[off:off + n]))[0, 1])
            off += n
        return np.asarray(out)

    base_ics = ics_of(stack_parts)

    surgery = {"grid": {}, "best": None}
    best = (float(base_ics.mean()), 0, 0.0, 0.0, 0.0)
    hist_tr = [_sec_dicts(tr, i) for i in range(len(tr["g"]))]
    for ahi in SURGERY_GRID["alpha_hi"]:
        rv = recurse_multi(mA, mB, va, hist_tr[-6:], alpha_hi=ahi)
        for K in SURGERY_GRID["K"]:
            for w0 in SURGERY_GRID["w0"]:
                for gamma in SURGERY_GRID["gamma"]:
                    parts = list(stack_parts)
                    for i in range(min(K, len(parts))):
                        w = min(w0 * gamma ** i, 1.0)
                        n = int(full_groups[i])
                        parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rv[i], [n])
                    m = float(ics_of(parts).mean())
                    surgery["grid"][f"K{K}_w{w0}_g{gamma}_a{ahi}"] = m
                    if m > best[0]:
                        best = (m, K, w0, gamma, ahi)
    best_m, K, w0, gamma, ahi = best
    surgery["best"] = {"K": K, "w0": w0, "gamma": gamma, "alpha_hi": ahi, "valid_ic": best_m}
    print(f"[exp023g] surgery best: K={K} w0={w0} gamma={gamma} alpha_hi={ahi} -> valid {best_m:+.6f} "
          f"(new stack {base_ics.mean():+.6f}, +{best_m - base_ics.mean():.6f})", flush=True)

    metrics = {
        "experiment": "exp_023g_catboost_stack_surgery",
        "stage": "fit",
        "catboost_standalone_full_valid_ic": cat_standalone_ic,
        "reblend_full_valid_ic": reblend_ic,
        "baseline_router_full_valid_ic": 0.091389,
        "capped_valid_mean_ic": capped_ic,
        "baseline_capped_valid_ic": 0.090487,
        "exp021_capped_valid_ic": 0.090972,
        "surgery": surgery,
        "surgery_base_stack_valid_ic": float(base_ics.mean()),
        "surgery_valid_ic": best_m,
        "elapsed_s": round(time.time() - t0, 1),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "metrics_fit.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[exp023g] fit stage DONE", flush=True)
    print(json.dumps({k: metrics[k] for k in ("capped_valid_mean_ic", "reblend_full_valid_ic",
                                              "catboost_standalone_full_valid_ic", "surgery_valid_ic")}, indent=2), flush=True)
    return metrics


# ---------------- submit 阶段 ----------------

def submit_stage(config: RunConfig, ctx: DataContext) -> dict:
    t0 = time.time()
    cap = config.stock_cap
    CKPT.mkdir(parents=True, exist_ok=True)
    head = MultiObjectiveRankHead(expert_count=6, state_dim=8)
    restore_torch_checkpoint(head, CKPT / "multi_objective_head.pt")
    base = torch.tensor([BASE_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    minimum = torch.tensor([MIN_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    router = StateRouter(8, len(FAMILIES), base, minimum)
    restore_torch_checkpoint(router, CKPT / "state_router.pt")
    print("[exp023g] head/router restored", flush=True)

    print("[exp023g] retrain final CatBoost tabular", flush=True)
    final_X, final_y, final_rel, final_groups = _combined_supervised_arrays(ctx, cap)
    models = train_cat_tabular(config, final_X, final_y, final_rel, final_groups)
    del final_y, final_rel, final_groups
    test_X, _, _, test_groups, _ = _tabular_arrays(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT)
    tabular_test, _ = predict_cat_tabular(models, test_X, test_groups)
    del test_X, models

    family = {}
    for name in FAMILIES:
        family[name] = tabular_test.astype(np.float32) if name == "tabular" else np.load(EXP016_FULL / f"family_{name}.npy").astype(np.float32)

    final_anchor_model = load_exp015_anchor(EXP016_CACHE / "checkpoints" / "final" / "anchor.txt")
    combined_anchor = predict_exp015_anchor(final_anchor_model, final_X)
    train_rows = int(_tabular_arrays(ctx, "train", 486, VALID_START, cap)[0].shape[0])
    train_anchor, valid_anchor = combined_anchor[:train_rows], combined_anchor[train_rows:]
    del final_X, combined_anchor
    raw = lambda: _combined_batch_factory(ctx, cap, train_anchor, valid_anchor)
    history, mask = _period_sample(raw)
    periods = fit_period_library(history, mask)

    test_anchor = load_exp015_test_anchor(ctx)
    family6 = {FAMILIES[i]: family[FAMILIES[i]] for i in range(6)}
    test_factory = lambda: _torch_batches(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT, test_anchor, periods)
    experts, _, states, groups = _collect_head_inputs(test_factory, family6)
    output = _head_outputs_by_group(head, experts, states, groups)
    family["multi_objective_rank"] = output["score"].numpy().astype(np.float32)

    router.to("cpu").eval()
    with torch.no_grad():
        confidence = torch.stack([part.mean() for part in torch.split(output["confidence"], groups.tolist())])
        weights = router(states, confidence)
    blended = dynamic_blend_family_predictions(family, groups.numpy(), weights.numpy())

    # 手术叠加：加载 fit 阶段参数与递归模型
    fit_metrics = json.loads((RESULT / "metrics_fit.json").read_text(encoding="utf-8"))
    surg = fit_metrics["surgery"]["best"]
    K, w0, gamma, ahi = surg["K"], surg["w0"], surg["gamma"], surg["alpha_hi"]

    import lightgbm as lgb
    mA = lgb.Booster(model_file=str(CKPT / "recursion_mA.txt"))
    mB = lgb.Booster(model_file=str(CKPT / "recursion_mB.txt"))
    tr = _load_split("train")
    va = _load_split("valid")
    te = _load_split("test")
    hist_va = [_sec_dicts(va, i) for i in range(len(va["g"]))]
    rec_te_parts = recurse_multi(mA, mB, te, hist_va[-6:], alpha_hi=ahi)

    # blended 向量按 common test 顺序切分（与 exp023e/f 同协议：grid->vector 往返对齐）
    te_time = np.load(C / "test_time.npy").astype(np.int32)
    te_stock = np.load(C / "test_stock.npy").astype(np.int32)
    grid_plain = vector_to_grid(blended.astype(np.float32), te_time, te_stock, TEST_START, TEST_STOP - TEST_START)
    stack_te = grid_plain[te_time - TEST_START, te_stock]
    parts = list(np.split(stack_te, np.cumsum(te["g"])[:-1]))
    assert parts[0].size == rec_te_parts[0].size, "test 截面切分与递归不一致"
    for i in range(min(K, len(parts))):
        w = min(w0 * gamma ** i, 1.0)
        n = int(te["g"][i])
        parts[i] = (1 - w) * group_rank(parts[i], [n]) + w * group_rank(rec_te_parts[i], [n])
    final = group_rank(np.concatenate(parts).astype(np.float32), te["g"])
    out = vector_to_grid(final, te_time, te_stock, TEST_START, TEST_STOP - TEST_START)
    mask_eval = np.zeros((TEST_STOP - TEST_START, STOCK_COUNT), dtype=bool)
    mask_eval[te_time - TEST_START, te_stock] = True
    contract = validate_prediction(out, mask_eval)
    np.save(RESULT / "prediction_1.npy", out)

    metrics = {
        "experiment": "exp_023g_catboost_stack_surgery",
        "stage": "submit",
        "surgery_applied": surg,
        "test_modified_sections": f"前 {K} 截面（3161..），其余为新栈（CatBoost tabular + 重训 head/router）",
        "contract": contract,
        "submission": "prediction_1.npy",
        "compliance": "第 t 截面仅用 X(<=t) 与 <t 给定标签/自有预测；CatBoost 60 轮来自 exp022 Phase A(55)×1.1 固定协议；无 t 后数据",
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("[exp023g] submit stage DONE", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps({
        "experiment": "exp_023g_catboost_stack_surgery",
        "task": "CatBoost tabular 专家替换 + head/router 重训 + 锚点手术",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_submission_overwritten": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage not in {"fit", "submit", "all"}:
        raise SystemExit("用法: run_exp023g.py [fit|submit|all]")
    config = RunConfig.from_environment()
    ctx = DataContext(load_sequence=True)
    if stage in {"fit", "all"}:
        fit_stage(config, ctx)
    if stage in {"submit", "all"}:
        submit_stage(config, ctx)


if __name__ == "__main__":
    main()
