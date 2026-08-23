from __future__ import annotations

"""exp_021：元头 + 路由一致重训（cat_5 原生类别 tabular 版）。

exp_020 只替换了 tabular 家族，head/router 仍按旧 tabular 训练。本实验在
OOF 六家族矩阵上用「categorical tabular」重算 states 后，端到端重训
multi_objective_head 与 state_router，并重新融合出 Test 提交。

复用（不重训）：
- 神经家族 checkpoint（OOF 折、official_valid、final）
- family_matrix.npy 中除 tabular 外的 5 个神经/锚点 OOF 列
- exp016 full 的 Test 家族预测（除 tabular）

重训：
- 3 个 OOF 折 + official_valid + final 共 5 套 tabular（LightGBM categorical）
- multi_objective_head、state_router

阶段：
- stage=fit    重训 tabular(OOF×3+official_valid) + head + router，capped valid 评估
- stage=submit 重训 final tabular，生成 Test 提交 prediction_1.npy
- stage=all    两者都做

训练保护：必须 DSCR_EXP016_MODE=full 且 DSCR_EXP016_ALLOW_TRAINING=YES。
输出写入 exp_021 自己的目录，不覆盖 exp016 / final_submission 任何产物。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")

from exp_016_unified_expert_fusion.config import (BASE_WEIGHTS, CACHE_DIR, FAMILIES, MIN_WEIGHTS,
                                                  OOF_FOLDS, RESULT_DIR, RunConfig, STOCK_COUNT,
                                                  TEST_START, TEST_STOP, VALID_START, VALID_STOP)
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
from exp_016_unified_expert_fusion.src.tabular_experts import predict_tabular_family, train_tabular_family
from exp_016_unified_expert_fusion.src.time_frequency import fit_period_library
from exp_016_unified_expert_fusion.src.training import restore_torch_checkpoint, train_head, train_router

EXP016_FULL = RESULT_DIR / "full"
EXP016_CACHE = CACHE_DIR
RESULT = Path(r"D:\google_dl\book\youanbei\04_results\exp_021_retrain_head_router")
CKPT = Path(r"D:\google_dl\book\youanbei\03_cache\exp_021_retrain_head_router")


def per_group_ic(pred, target, groups):
    scores, offset = [], 0
    for size in groups:
        size = int(size)
        scores.append(rank_ic(pred[offset:offset + size], target[offset:offset + size]))
        offset += size
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite))


def train_fold_tabular(config, ctx, split, start, stop, cap):
    X, y, rel, groups, _ = _tabular_arrays(ctx, split, start, stop, cap)
    models = train_tabular_family(config, X, y, rel, groups)
    return models, X, groups


def fit_stage(config: RunConfig, ctx: DataContext) -> dict:
    t0 = time.time()
    cap = config.stock_cap
    family_matrix = np.load(EXP016_CACHE / "oof_predictions" / "family_matrix.npy").astype(np.float32)

    experts_parts, target_parts, state_parts, group_parts = [], [], [], []
    offset = 0
    for fold_name, ts, te, ps, pe in OOF_FOLDS:
        print(f"[exp021] fold {fold_name}: retrain tabular", flush=True)
        models, train_X, train_groups = train_fold_tabular(config, ctx, "train", ts, te, cap)
        anchor_model = load_exp015_anchor(EXP016_CACHE / "checkpoints" / fold_name / "anchor.txt")
        train_anchor = predict_exp015_anchor(anchor_model, train_X)
        pred_X, pred_y, _, pred_groups, _ = _tabular_arrays(ctx, "train", ps, pe, cap)
        pred_anchor = predict_exp015_anchor(anchor_model, pred_X)
        tabular_pred, _ = predict_tabular_family(models, pred_X, pred_groups)
        del train_X, pred_X

        raw_factory = lambda ts2=ts, te2=te, av=train_anchor: _torch_batches(ctx, "train", ts2, te2, cap, av)
        history, mask = _period_sample(raw_factory)
        periods = fit_period_library(history, mask)

        rows = tabular_pred.size
        old_six = family_matrix[offset:offset + rows].copy()
        new_six = old_six
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
    print(f"[exp021] OOF rows={experts.shape[0]}, groups={groups.numel()}, elapsed={time.time()-t0:.1f}s", flush=True)

    # 重训 head
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

    # head 输出 → 第 7 家族 + desired 权重 → 重训 router
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

    # capped valid 评估（与 exp016 metadata 0.090487 同口径）
    valid_anchor_model = load_exp015_anchor(EXP016_CACHE / "checkpoints" / "official_valid" / "anchor.txt")
    valid_X, valid_y, _, valid_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, cap)
    valid_anchor = predict_exp015_anchor(valid_anchor_model, valid_X)
    print("[exp021] retrain official_valid tabular", flush=True)
    models_val, _, _ = train_fold_tabular(config, ctx, "train", 486, VALID_START, cap)
    tabular_valid, _ = predict_tabular_family(models_val, valid_X, valid_groups)
    del valid_X

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

    metrics = {
        "experiment": "exp_021_retrain_head_router",
        "stage": "fit",
        "oof_rows": int(experts.shape[0]),
        "capped_valid_mean_ic": capped_ic,
        "baseline_capped_valid_ic": 0.09048734,
        "capped_valid_delta": round(capped_ic - 0.09048734, 6),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("[exp021] fit stage DONE", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def submit_stage(config: RunConfig, ctx: DataContext, fit_metrics: dict | None = None) -> dict:
    t0 = time.time()
    cap = config.stock_cap
    CKPT.mkdir(parents=True, exist_ok=True)
    head = MultiObjectiveRankHead(expert_count=6, state_dim=8)
    restore_torch_checkpoint(head, CKPT / "multi_objective_head.pt")
    base = torch.tensor([BASE_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    minimum = torch.tensor([MIN_WEIGHTS[name] for name in FAMILIES], dtype=torch.float32)
    router = StateRouter(8, len(FAMILIES), base, minimum)
    restore_torch_checkpoint(router, CKPT / "state_router.pt")
    print("[exp021] head/router restored", flush=True)

    # final tabular（categorical 版）
    print("[exp021] retrain final tabular", flush=True)
    final_X, final_y, final_rel, final_groups = _combined_supervised_arrays(ctx, cap)
    models = train_tabular_family(config, final_X, final_y, final_rel, final_groups)
    del final_y, final_rel, final_groups
    test_X, _, _, test_groups, _ = _tabular_arrays(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT)
    tabular_test, _ = predict_tabular_family(models, test_X, test_groups)
    del test_X

    # 7 列 family：新 tabular + exp016 其它家族
    family = {}
    for name in FAMILIES:
        family[name] = tabular_test.astype(np.float32) if name == "tabular" else np.load(EXP016_FULL / f"family_{name}.npy").astype(np.float32)

    # periods：final 训练侧拟合
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
    grid = vector_to_grid(blended, ctx.common["test"]["time"], ctx.common["test"]["stock"], TEST_START, TEST_STOP - TEST_START)
    contract = validate_prediction(grid, ctx.test_evaluation_mask())
    RESULT.mkdir(parents=True, exist_ok=True)
    np.save(RESULT / "prediction_1.npy", grid)

    metrics = {
        "experiment": "exp_021_retrain_head_router",
        "stage": "submit",
        "contract": contract,
        "submission": "prediction_1.npy",
        "fit_reference": fit_metrics,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("[exp021] submit stage DONE", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if stage not in {"fit", "submit", "all"}:
        raise SystemExit("用法: run_exp021.py [fit|submit|all]")
    config = RunConfig.from_environment()
    ctx = DataContext(load_sequence=True)
    fit_metrics = None
    if stage in {"fit", "all"}:
        fit_metrics = fit_stage(config, ctx)
        RESULT.mkdir(parents=True, exist_ok=True)
        (RESULT / "metrics_fit.json").write_text(json.dumps(fit_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if stage in {"submit", "all"}:
        submit_metrics = submit_stage(config, ctx, fit_metrics)
        (RESULT / "metrics.json").write_text(json.dumps(submit_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (RESULT / "metadata.json").write_text(json.dumps({
            "experiment": "exp_021_retrain_head_router",
            "task": "head+router 一致重训（categorical tabular）",
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_submission_overwritten": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
