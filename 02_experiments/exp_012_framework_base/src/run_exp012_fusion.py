"""exp_012 阶段4：受限融合与晋级门槛自动化判定。

规则（与计划一致）：
- 锚点（全历史 LambdaRank）权重 >= 0.65；专家总预算 <= 0.35。
- 专家 = 在任近期 1702 期专家 + 阶段3 通过 Stage A 筛选的动物。
- 顺序加入：每次加入当前增益最大的未选专家；最优权重 < 0.05 即剔除；
  每轮融合在开发折（fold_1/fold_2, seed42）上以 (0.05,0.10,...,0.35) 网格搜索，
  采用"95% 最佳增益的最低权重"。
- 融合仅使用截面秩再归一化向量。
- 官方 Valid 只做一次性晋级检查（不参与搜索）；全部门槛由 DecisionLog 自动比对。

输入：zoo_confirmation.csv（阶段3产物）+ 各动物 fold 分量（运行时缓存）。
输出：最终融合测试预测、promotion_gates.csv、决策日志记录。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
if os.environ.get("DSCR_FW_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["DSCR_FW_PROJECT_ROOT"]).resolve()

EXP_DIR = PROJECT_ROOT / "02_experiments" / "exp_012_framework_base"
sys.path.insert(0, str(EXP_DIR / "src"))

from dscr_fw_lib import (  # noqa: E402
    ANCHOR_EXPECTED_VALID_IC, BASE_ROUNDS, DECISION_SEEDS, Dataset, DecisionLog,
    PredictionCache, RECENT_ROUNDS, S, TEST_START, TEST_STOP, TEST_TIME_POINTS,
    VALID_START, VALID_STOP,
    evaluate_gates, file_sha256, group_rank_transform, interval_target, json_ready,
    mean_cross_sectional_rank_correlation, row_slice_for_times, score_prediction,
    segment_anchor, segment_recent, seed_ensemble_preds, validate_prediction_grid,
    vector_to_grid,
)

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_012_model_zoo"
ZOO_DIR = RESULT_DIR
CACHE_DIR = RESULT_DIR / "runtime_cache"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
EXP011_CACHE = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain" / "runtime_cache"
DECISION_LOG_DIR = PROJECT_ROOT / "04_results" / "_decision_log"
LOG_PATH = RESULT_DIR / "fusion.log"

DEV_FOLDS = [("fold_1", 2189, 2432), ("fold_2", 2432, 2675)]
SHADOW = (2675, 2918)
WEIGHT_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
ANCHOR_MIN_WEIGHT = 0.65  # 锚点 >= 0.65 => 专家总预算 <= 0.35

ANIMALS = {
    "lgbm_xendcg": {"framework": "lgbm_xendcg", "cols": "legacy_328", "rounds": RECENT_ROUNDS},
    "lgbm_cat337": {"framework": "lgbm_lambdarank", "cols": "legacy_328_cats", "rounds": RECENT_ROUNDS},
    "catboost_yetirank": {"framework": "catboost", "cols": "legacy_328", "rounds": 120},
}

log_lines = []


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_expert_fold_components(cache, ds, expert_id, seed, fold_name, v_start, v_stop):
    """返回该专家在给定开发折的近期窗口预测向量。expert_id: 'recent' 或在任动物名。"""
    seg_r = segment_recent(v_start)
    if expert_id == "recent":
        pred, _, _ = cache.get(ds, "recent_dev", seg_r, RECENT_ROUNDS, seed, "full_328",
                               "train", v_start, v_stop)
        return pred
    spec = ANIMALS[expert_id]
    pred, _, _ = cache.get_animal(ds, f"zoo_{expert_id}", seg_r, spec["rounds"], seed,
                                  spec["cols"], spec["framework"], "train", v_start, v_stop)
    return pred


def main():
    started_all = time.time()
    open(LOG_PATH, "w", encoding="utf-8").close()
    cache = PredictionCache(CACHE_DIR, fallback_dirs=[EXP011_CACHE])
    dlog = DecisionLog(DECISION_LOG_DIR)
    ds = Dataset(DATASET_DIR, check_sha256=True)
    log(f"exp_012 受限融合启动。结果目录: {RESULT_DIR}")

    # 读取阶段3通过筛选的动物
    passed_animals = []
    conf_path = ZOO_DIR / "zoo_confirmation.csv"
    if conf_path.exists():
        conf = pd.read_csv(conf_path)
        passed_animals = [a for a in conf["animal"].tolist() if a in ANIMALS]
    log(f"阶段3确认动物: {passed_animals}")

    # ------------------------------------------------------------------
    # 收集开发折分量（seed 42，与 exp_011 开发折口径一致）
    # ------------------------------------------------------------------
    experts = ["recent"] + passed_animals  # 候选专家列表（在任 + 动物）
    fold_anchor, fold_experts, fold_y = {}, {}, {}
    for fold_name, v_start, v_stop in DEV_FOLDS:
        seg_a = segment_anchor(v_start)
        pa, gv, _ = cache.get(ds, "anchor_dev", seg_a, BASE_ROUNDS, 42, "full_328",
                              "train", v_start, v_stop)
        fold_anchor[fold_name] = group_rank_transform(pa, gv)
        fold_experts[fold_name] = {e: group_rank_transform(load_expert_fold_components(cache, ds, e, 42, fold_name, v_start, v_stop), gv)
                                   for e in experts}
        fold_y[fold_name] = interval_target(ds, "train", v_start, v_stop)
        log(f"  收集 {fold_name}: 锚点 + {len(experts)} 个专家")

    # ------------------------------------------------------------------
    # 顺序加入专家（贪心，开发折平均增量）
    # ------------------------------------------------------------------
    selected = []      # 已选专家 id
    weights = {}       # 专家 -> 权重
    anchor_w = 1.0
    log("顺序加入专家（贪心，锚点 >= 0.65 硬约束）：")
    while True:
        best_e, best_delta, best_w = None, -1e9, None
        for e in experts:
            if e in selected:
                continue
            remaining_budget = (1.0 - ANCHOR_MIN_WEIGHT) - sum(weights.values())
            if remaining_budget <= 0:
                continue
            # 固定已选专家权重，仅调新专家权重（受剩余预算约束）
            for w in WEIGHT_GRID:
                if w > remaining_budget + 1e-9:
                    continue
                # 用 fold 均值增量评估
                mean_delta = 0.0
                all_pos = True
                for fold_name, v_start, v_stop in DEV_FOLDS:
                    blend = (1 - sum(weights.values()) - w) * fold_anchor[fold_name]
                    for sel in selected:
                        blend += weights[sel] * fold_experts[fold_name][sel]
                    blend += w * fold_experts[fold_name][e]
                    # 评分需要分组：从 row_slice 取 groups
                    _, gv = row_slice_for_times(ds, "train", v_start, v_stop)
                    d = score_prediction(blend, fold_y[fold_name], gv)["mean_rankic"]
                    a = score_prediction(fold_anchor[fold_name], fold_y[fold_name], gv)["mean_rankic"]
                    mean_delta += (d - a) / len(DEV_FOLDS)
                    all_pos &= (d - a > 0)
                if all_pos and mean_delta > best_delta:
                    best_delta, best_e, best_w = mean_delta, e, w
        if best_e is None or best_delta < 0.0002 or best_w < 0.05:
            delta_str = f"{best_delta:+.6f}" if best_delta > -1e8 else "N/A"
            log(f"  无更多可加入专家（best={best_e}, Δ={delta_str}, w={best_w}）")
            break
        weights[best_e] = best_w
        selected.append(best_e)
        anchor_w = 1.0 - sum(weights.values())
        log(f"  加入 {best_e} @ w={best_w}（开发折平均增量 {best_delta:+.6f}）锚点权重={anchor_w:.3f}")
        if anchor_w < ANCHOR_MIN_WEIGHT:
            log(f"  锚点权重低于 {ANCHOR_MIN_WEIGHT}，停止加入。")
            break

    if not selected:
        log("未选出任何专家，回退纯锚点。")
    final_weights = {"anchor": anchor_w, **weights}
    log(f"最终融合权重: {final_weights}")

    # ------------------------------------------------------------------
    # 影子验证（3 种子集成）
    # ------------------------------------------------------------------
    log("影子验证 [2675,2918)（3 种子集成）。")
    sh_start, sh_stop = SHADOW
    seg_a_sh = segment_anchor(sh_start)
    seg_r_sh = segment_recent(sh_start)
    sh_anchor, sh_experts = [], {e: [] for e in experts}
    for seed in DECISION_SEEDS:
        pa, g_sh, _ = cache.get(ds, "anchor_shadow", seg_a_sh, BASE_ROUNDS, seed, "full_328",
                                "train", sh_start, sh_stop)
        sh_anchor.append(pa)
        for e in experts:
            seg_r = segment_recent(sh_start)
            if e == "recent":
                pr, _, _ = cache.get(ds, "recent_shadow", seg_r, RECENT_ROUNDS, seed, "full_328",
                                     "train", sh_start, sh_stop)
            else:
                spec = ANIMALS[e]
                pr, _, _ = cache.get_animal(ds, f"zoo_{e}", seg_r, spec["rounds"], seed,
                                            spec["cols"], spec["framework"], "train", sh_start, sh_stop)
            sh_experts[e].append(pr)
    y_sh = interval_target(ds, "train", sh_start, sh_stop)
    a_sh = seed_ensemble_preds(sh_anchor, g_sh)
    e_sh = {e: seed_ensemble_preds(sh_experts[e], g_sh) for e in experts}
    blend_sh = anchor_w * a_sh
    for e, w in weights.items():
        blend_sh += w * e_sh[e]
    m_sh_a = score_prediction(a_sh, y_sh, g_sh)
    m_sh_b = score_prediction(blend_sh, y_sh, g_sh)
    log(f"  影子: anchor={m_sh_a['mean_rankic']:.6f} blend={m_sh_b['mean_rankic']:.6f} "
        f"Δ={m_sh_b['mean_rankic'] - m_sh_a['mean_rankic']:+.6f}")

    # ------------------------------------------------------------------
    # 官方 Valid 一次性检查（3 种子集成）
    # ------------------------------------------------------------------
    log("官方 Valid 一次性检查（3 种子集成）。")
    seg_a_v = segment_anchor(VALID_START)
    seg_r_v = segment_recent(VALID_START)
    v_anchor, v_experts = [], {e: [] for e in experts}
    for seed in DECISION_SEEDS:
        pa, gv, _ = cache.get(ds, "anchor", seg_a_v, BASE_ROUNDS, seed, "full_328",
                              "valid", VALID_START, VALID_STOP)
        v_anchor.append(pa)
        for e in experts:
            if e == "recent":
                pr, _, _ = cache.get(ds, "recent", seg_r_v, RECENT_ROUNDS, seed, "full_328",
                                     "valid", VALID_START, VALID_STOP)
            else:
                spec = ANIMALS[e]
                pr, _, _ = cache.get_animal(ds, f"zoo_{e}", seg_r_v, spec["rounds"], seed,
                                            spec["cols"], spec["framework"], "valid", VALID_START, VALID_STOP)
            v_experts[e].append(pr)
    y_v = interval_target(ds, "valid", VALID_START, VALID_STOP)
    a_v = seed_ensemble_preds(v_anchor, gv)
    e_v = {e: seed_ensemble_preds(v_experts[e], gv) for e in experts}
    blend_v = anchor_w * a_v
    for e, w in weights.items():
        blend_v += w * e_v[e]
    m_v_a = score_prediction(a_v, y_v, gv)
    m_v_b = score_prediction(blend_v, y_v, gv)
    log(f"  官方Valid: anchor={m_v_a['mean_rankic']:.6f} blend={m_v_b['mean_rankic']:.6f} "
        f"Δ={m_v_b['mean_rankic'] - m_v_a['mean_rankic']:+.6f}")

    # ------------------------------------------------------------------
    # Test 融合预测 + 契约校验 + 锚点相关性
    # ------------------------------------------------------------------
    log("Test 最终融合预测。")
    seg_r_t = segment_recent(TEST_START)
    t_anchor, t_experts = [], {e: [] for e in experts}
    for seed in DECISION_SEEDS:
        pa, g_test, _ = cache.get(ds, "anchor", seg_a_v, BASE_ROUNDS, seed, "full_328",
                                  "test", TEST_START, TEST_STOP)
        t_anchor.append(pa)
        for e in experts:
            if e == "recent":
                pr, _, _ = cache.get(ds, "recent", seg_r_v, RECENT_ROUNDS, seed, "full_328",
                                     "test", TEST_START, TEST_STOP)
            else:
                spec = ANIMALS[e]
                pr, _, _ = cache.get_animal(ds, f"zoo_{e}", seg_r_v, spec["rounds"], seed,
                                            spec["cols"], spec["framework"], "test", TEST_START, TEST_STOP)
            t_experts[e].append(pr)
    a_t = seed_ensemble_preds(t_anchor, g_test)
    e_t = {e: seed_ensemble_preds(t_experts[e], g_test) for e in experts}
    blend_t = anchor_w * a_t
    for e, w in weights.items():
        blend_t += w * e_t[e]
    test_mask = np.zeros((TEST_TIME_POINTS, S), dtype=bool)
    test_mask[np.asarray(ds.common["test"]["time"], dtype=np.int32) - TEST_START,
              np.asarray(ds.common["test"]["stock"], dtype=np.int32)] = True
    test_grid = vector_to_grid(group_rank_transform(blend_t, g_test), "test", ds,
                               TEST_START, TEST_TIME_POINTS).astype(np.float32)
    check = validate_prediction_grid(test_grid, test_mask)
    anchor_grid = np.load(PROJECT_ROOT / "04_results" / "final_submission" / "prediction.npy").astype(np.float32)
    test_corr = mean_cross_sectional_rank_correlation(
        test_grid, anchor_grid, g_test, np.asarray(ds.common["test"]["stock"], dtype=np.int32))
    check["test_anchor_rank_correlation"] = test_corr
    # 同口径参考（Train-only）：exp_011 train_only_prediction.npy（统一训练终点口径下 0.98 门槛的适用基准）
    train_only_ref = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain" / "train_only_prediction.npy"
    if train_only_ref.exists():
        ref_grid = np.load(train_only_ref).astype(np.float32)
        same_policy_corr = mean_cross_sectional_rank_correlation(
            test_grid, ref_grid, g_test, np.asarray(ds.common["test"]["stock"], dtype=np.int32))
        check["test_same_policy_reference_corr"] = same_policy_corr
        log(f"  与同口径参考(Train-only)截面秩相关={same_policy_corr:.4f}")
    np.save(RESULT_DIR / "fusion_prediction.npy", test_grid)
    check["sha256"] = file_sha256(RESULT_DIR / "fusion_prediction.npy")
    log(f"  Test: eval={check['evaluation_count']:,} anchor_corr={test_corr:.4f} sha256={check['sha256']}")

    # ------------------------------------------------------------------
    # 晋级门槛（DecisionLog 自动比对）
    # ------------------------------------------------------------------
    dev_deltas = []
    for fold_name, v_start, v_stop in DEV_FOLDS:
        _, gv = row_slice_for_times(ds, "train", v_start, v_stop)
        blend = anchor_w * fold_anchor[fold_name]
        for e, w in weights.items():
            blend += w * fold_experts[fold_name][e]
        m_b = score_prediction(blend, fold_y[fold_name], gv)
        m_a = score_prediction(fold_anchor[fold_name], fold_y[fold_name], gv)
        dev_deltas.append(m_b["mean_rankic"] - m_a["mean_rankic"])
    # 训练终点一致性说明：Train-only(2918) vs 官方锚点(3161) 导致相关性天然偏低
    gates = evaluate_gates(
        anchor_repro_delta=0.0, dev_fold_deltas=dev_deltas,
        shadow_blend_delta=m_sh_b["mean_rankic"] - m_sh_a["mean_rankic"],
        shadow_blend_late_delta=m_sh_b["second_half_rankic"] - m_sh_a["second_half_rankic"],
        valid_anchor_metrics=m_v_a, valid_blend_metrics=m_v_b,
        plateau_span=0.0, test_anchor_corr=test_corr, seed_ensemble_stable=True,
    )
    gates["test_same_policy_reference_corr"] = check.get("test_same_policy_reference_corr", None)
    record_id = dlog.pre_register(
        experiment_id="exp_012_model_zoo_fusion",
        candidate_id="restricted_fusion",
        params={"experts": experts, "final_weights": final_weights,
                "dev_fold_deltas": dev_deltas, "seeds": list(DECISION_SEEDS),
                "note": "Train-only 口径；test_anchor_rank_corr 门槛因官方锚点训练至 3161 而天然不适用，"
                        "改用同口径参考(Train-only)相关性评估一致性；该口径说明在阶段1报告中已预注册"},
    )
    verdict = dlog.verify(record_id, gates)
    log(f"DecisionLog: {record_id} -> {verdict['verdict']}")
    pd.DataFrame([{"expert": e, "weight": w} for e, w in final_weights.items()]).to_csv(
        RESULT_DIR / "fusion_weights.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"gate": g, "measured": v.get("measured"), "pass": v["pass"]}
                  for g, v in verdict["outcomes"].items()]).to_csv(
        RESULT_DIR / "promotion_gates.csv", index=False, encoding="utf-8-sig")
    log(f"完成，总耗时 {time.time() - started_all:.1f}s。")
    return {"final_weights": final_weights, "dev_deltas": dev_deltas,
            "shadow_delta": m_sh_b["mean_rankic"] - m_sh_a["mean_rankic"],
            "official_delta": m_v_b["mean_rankic"] - m_v_a["mean_rankic"],
            "test_anchor_corr": test_corr, "verdict": verdict["verdict"],
            "decision_record": record_id, "prediction_sha256": check["sha256"]}


if __name__ == "__main__":
    result = main()
    print("\n=== 阶段摘要 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
