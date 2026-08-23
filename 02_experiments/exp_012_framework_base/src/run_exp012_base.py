"""exp_012 统一运行时地基：骨架跑通。

流程：数据契约复验 -> 锚点复现（0.092940 容差 3e-4）-> 近期 1702 期专家 ->
开发折权重搜索 -> 影子验证 -> 官方 Valid 一次性检查 -> 最终测试预测 + 提交契约
-> DecisionLog 预注册/自动比对/归档。

预测优先从 exp_011 runtime_cache 复用（指纹一致零成本），缺失时重新训练并写入
本实验 primary cache。本骨架同时是对统一运行时的正确性验证：所有指标应与
exp_011 记录逐项一致。
"""
from __future__ import annotations

import gc
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
    ANCHOR_EXPECTED_VALID_IC, BASE_ROUNDS, DEV_FOLDS, DECISION_SEEDS, Dataset,
    DecisionLog, FOLD_SPECS, GATE_SPECS, PredictionCache, RECENT_LOOKBACK,
    RECENT_ROUNDS, S, TEST_START, TEST_STOP, TEST_TIME_POINTS, TRAIN_START,
    VALID_START, VALID_STOP, WEIGHT_CANDIDATES,
    evaluate_gates, file_sha256, group_rank_transform, interval_target,
    json_ready, mean_cross_sectional_rank_correlation,
    rank_ic_self_test, row_slice_for_times, score_prediction,
    segment_anchor, segment_recent, seed_ensemble_preds,
    validate_prediction_grid, vector_to_grid,
)

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_012_framework_base"
CACHE_DIR = RESULT_DIR / "runtime_cache"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
ANCHOR_PATH = PROJECT_ROOT / "04_results" / "final_submission" / "prediction.npy"
EXP011_CACHE = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain" / "runtime_cache"
DECISION_LOG_DIR = PROJECT_ROOT / "04_results" / "_decision_log"
LOG_PATH = RESULT_DIR / "run.log"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def test_mask_of(ds):
    test_mask = np.zeros((TEST_TIME_POINTS, S), dtype=bool)
    test_mask[
        np.asarray(ds.common["test"]["time"], dtype=np.int32) - TEST_START,
        np.asarray(ds.common["test"]["stock"], dtype=np.int32),
    ] = True
    return test_mask


def main():
    started_all = time.time()
    open(LOG_PATH, "w", encoding="utf-8").close()
    cache = PredictionCache(CACHE_DIR, fallback_dirs=[EXP011_CACHE])
    dlog = DecisionLog(DECISION_LOG_DIR)
    log(f"exp_012 framework base 启动。结果目录: {RESULT_DIR}")

    # =====================================================================
    # 阶段 0.1：数据契约复验
    # =====================================================================
    log("阶段0.1：数据契约复验（READY / manifest SHA-256 / 形状 / 测试掩码）。")
    ds = Dataset(DATASET_DIR, check_sha256=True)
    test_mask = test_mask_of(ds)
    assert int(test_mask.sum()) == 2_042_538
    rank_ic_self_test()
    log(f"数据契约通过：test_mask 评价位 {int(test_mask.sum()):,}；RankIC 自检通过。")

    # =====================================================================
    # 阶段 0.2：锚点复现（官方 Valid，训练至 2918，seed42）
    # =====================================================================
    log("阶段0.2：锚点复现 [486,2918) r8 s42 -> 官方 Valid。")
    seg_a = segment_anchor(VALID_START)
    pa, ga, _ = cache.get(ds, "anchor", seg_a, BASE_ROUNDS, 42, "full_328",
                          "valid", VALID_START, VALID_STOP)
    y_valid = interval_target(ds, "valid", VALID_START, VALID_STOP)
    ma = score_prediction(pa, y_valid, ga)
    anchor_delta = ma["mean_rankic"] - ANCHOR_EXPECTED_VALID_IC
    repro_ok = abs(anchor_delta) <= 0.0003
    log(f"锚点 Valid RankIC: {ma['mean_rankic']:.6f} (期望 {ANCHOR_EXPECTED_VALID_IC:.6f}) "
        f"复现判定: {'通过' if repro_ok else '未通过'}")
    assert repro_ok, "锚点复现失败，统一运行时不可用"

    # =====================================================================
    # 阶段 0.3：近期 1702 期专家（官方 Valid + Test）
    # =====================================================================
    log("阶段0.3：近期专家 [1216,2918) r16 s42 -> Valid + Test。")
    seg_r = segment_recent(VALID_START)
    pr, _, _ = cache.get(ds, "recent", seg_r, RECENT_ROUNDS, 42, "full_328",
                         "valid", VALID_START, VALID_STOP)
    pr_test, gr_test, _ = cache.get(ds, "recent", seg_r, RECENT_ROUNDS, 42, "full_328",
                                    "test", TEST_START, TEST_STOP)
    mr = score_prediction(pr, y_valid, ga)
    log(f"近期专家 Valid RankIC: {mr['mean_rankic']:.6f} (exp_011: 0.089657)")

    # =====================================================================
    # 阶段 0.4：三级验证骨架
    # =====================================================================
    log("阶段0.4：开发折权重搜索（fold_1/fold_2, seed42）。")
    fold_rows = []
    fold_preds = {}
    for fold_name in DEV_FOLDS:
        spec = FOLD_SPECS[fold_name]
        v_start, v_stop = spec["predict"]
        seg_a_f = segment_anchor(v_start)
        seg_r_f = segment_recent(v_start)
        pa_f, gv_f, _ = cache.get(ds, "anchor_dev", seg_a_f, BASE_ROUNDS, 42, "full_328",
                                  "train", v_start, v_stop)
        pr_f, _, _ = cache.get(ds, "recent_dev", seg_r_f, RECENT_ROUNDS, 42, "full_328",
                               "train", v_start, v_stop)
        fold_preds[fold_name] = (pa_f, pr_f, gv_f)
        y_f = interval_target(ds, "train", v_start, v_stop)
        m_a = score_prediction(pa_f, y_f, gv_f)
        m_r = score_prediction(pr_f, y_f, gv_f)
        a_rank = group_rank_transform(pa_f, gv_f)
        r_rank = group_rank_transform(pr_f, gv_f)
        fold_rows.append({"fold": fold_name, "model": "anchor", "rankic": m_a["mean_rankic"]})
        fold_rows.append({"fold": fold_name, "model": "recent_expert", "rankic": m_r["mean_rankic"]})
        for w in WEIGHT_CANDIDATES:
            blend = (1 - w) * a_rank + w * r_rank
            mb = score_prediction(blend, y_f, gv_f)
            fold_rows.append({"fold": fold_name, "model": "blend", "weight": w,
                              "rankic": mb["mean_rankic"]})
        log(f"  {fold_name}: anchor={m_a['mean_rankic']:.6f} recent={m_r['mean_rankic']:.6f}")
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(RESULT_DIR / "fold_results.csv", index=False, encoding="utf-8-sig")

    weight_rows = []
    for w in WEIGHT_CANDIDATES:
        sub = fold_df[(fold_df["model"] == "blend") & (fold_df["weight"] == w)]
        m1 = float(sub[sub["fold"] == "fold_1"]["rankic"].iloc[0])
        m2 = float(sub[sub["fold"] == "fold_2"]["rankic"].iloc[0])
        a1 = float(fold_df[(fold_df["fold"] == "fold_1") & (fold_df["model"] == "anchor")]["rankic"].iloc[0])
        a2 = float(fold_df[(fold_df["fold"] == "fold_2") & (fold_df["model"] == "anchor")]["rankic"].iloc[0])
        weight_rows.append({"weight": w, "fold_1_delta": m1 - a1, "fold_2_delta": m2 - a2,
                            "mean_delta": ((m1 - a1) + (m2 - a2)) / 2})
    wdf = pd.DataFrame(weight_rows)
    best_delta = float(wdf["mean_delta"].max())
    near = wdf[wdf["mean_delta"] >= best_delta * 0.95].sort_values("weight")
    selected_w = float(near.iloc[0]["weight"])
    wdf["selected"] = wdf["weight"] == selected_w
    wdf.to_csv(RESULT_DIR / "weight_search.csv", index=False, encoding="utf-8-sig")
    dev_deltas = [float(wdf[wdf["weight"] == selected_w]["fold_1_delta"].iloc[0]),
                  float(wdf[wdf["weight"] == selected_w]["fold_2_delta"].iloc[0])]
    log(f"权重选择: 最佳平均增量={best_delta:.6f} -> 选定 w={selected_w} ({dev_deltas})")

    # 影子验证 [2675,2918)（3 种子集成）
    log("阶段0.4b：影子验证 [2675,2918) 3 种子集成。")
    sh_start, sh_stop = FOLD_SPECS["shadow"]["predict"]
    seg_a_sh = segment_anchor(sh_start)
    seg_r_sh = segment_recent(sh_start)
    a_sh, r_sh = [], []
    for seed in DECISION_SEEDS:
        pa_s, g_sh, _ = cache.get(ds, "anchor_shadow", seg_a_sh, BASE_ROUNDS, seed, "full_328",
                                  "train", sh_start, sh_stop)
        pr_s, _, _ = cache.get(ds, "recent_shadow", seg_r_sh, RECENT_ROUNDS, seed, "full_328",
                               "train", sh_start, sh_stop)
        a_sh.append(pa_s)
        r_sh.append(pr_s)
    y_sh = interval_target(ds, "train", sh_start, sh_stop)
    a_sh_ens = seed_ensemble_preds(a_sh, g_sh)
    r_sh_ens = seed_ensemble_preds(r_sh, g_sh)
    m_sh_a = score_prediction(a_sh_ens, y_sh, g_sh)
    blend_sh = (1 - selected_w) * a_sh_ens + selected_w * r_sh_ens
    m_sh_b = score_prediction(blend_sh, y_sh, g_sh)
    shadow_delta = m_sh_b["mean_rankic"] - m_sh_a["mean_rankic"]
    shadow_late_delta = m_sh_b["second_half_rankic"] - m_sh_a["second_half_rankic"]
    log(f"影子: anchor={m_sh_a['mean_rankic']:.6f} blend={m_sh_b['mean_rankic']:.6f} "
        f"增量={shadow_delta:+.6f} 后段增量={shadow_late_delta:+.6f}")

    # 官方 Valid 一次性检查（3 种子集成）
    log("阶段0.4c：官方 Valid 一次性检查（3 种子集成）。")
    a_v, r_v = [], []
    for seed in DECISION_SEEDS:
        pa_v, gv, _ = cache.get(ds, "anchor", seg_a, BASE_ROUNDS, seed, "full_328",
                                "valid", VALID_START, VALID_STOP)
        pr_v, _, _ = cache.get(ds, "recent", seg_r, RECENT_ROUNDS, seed, "full_328",
                               "valid", VALID_START, VALID_STOP)
        a_v.append(pa_v)
        r_v.append(pr_v)
    a_v_ens = seed_ensemble_preds(a_v, gv)
    r_v_ens = seed_ensemble_preds(r_v, gv)
    m_v_a = score_prediction(a_v_ens, y_valid, gv)
    blend_v = (1 - selected_w) * a_v_ens + selected_w * r_v_ens
    m_v_b = score_prediction(blend_v, y_valid, gv)
    log(f"官方Valid: anchor={m_v_a['mean_rankic']:.6f} blend={m_v_b['mean_rankic']:.6f} "
        f"增量={m_v_b['mean_rankic'] - m_v_a['mean_rankic']:+.6f}")

    # =====================================================================
    # 阶段 0.5：最终测试预测 + 提交契约
    # =====================================================================
    log("阶段0.5：最终测试预测（Train-only 融合）与契约校验。")
    a_t, r_t = [], []
    for seed in DECISION_SEEDS:
        pa_t, g_test, _ = cache.get(ds, "anchor", seg_a, BASE_ROUNDS, seed, "full_328",
                                    "test", TEST_START, TEST_STOP)
        pr_t, _, _ = cache.get(ds, "recent", seg_r, RECENT_ROUNDS, seed, "full_328",
                               "test", TEST_START, TEST_STOP)
        a_t.append(pa_t)
        r_t.append(pr_t)
    a_t_ens = seed_ensemble_preds(a_t, g_test)
    r_t_ens = seed_ensemble_preds(r_t, g_test)
    blend_t = (1 - selected_w) * a_t_ens + selected_w * r_t_ens
    test_grid = vector_to_grid(group_rank_transform(blend_t, g_test), "test", ds,
                               TEST_START, TEST_TIME_POINTS).astype(np.float32)
    check = validate_prediction_grid(test_grid, test_mask)
    anchor_grid = np.load(ANCHOR_PATH).astype(np.float32)
    test_corr = mean_cross_sectional_rank_correlation(
        test_grid, anchor_grid, g_test, np.asarray(ds.common["test"]["stock"], dtype=np.int32))
    check["test_anchor_rank_correlation"] = test_corr
    check["sha256"] = None
    np.save(RESULT_DIR / "prediction.npy", test_grid)
    check["sha256"] = file_sha256(RESULT_DIR / "prediction.npy")
    log(f"最终预测: eval={check['evaluation_count']:,} non_eval={check['non_evaluation_count']:,} "
        f"anchor_corr={test_corr:.4f} sha256={check['sha256']}")

    # =====================================================================
    # 阶段 0.6：DecisionLog（预注册 + 自动比对 + 归档）
    # =====================================================================
    plateau_span = float(wdf[wdf["weight"] == selected_w]["mean_delta"].iloc[0] -
                         wdf["mean_delta"].min())
    ens_stable = bool(m_v_b["mean_rankic"] >= 0.093634 - 0.0005)  # 与 exp_011 同口径近似
    gates_measured = evaluate_gates(
        anchor_repro_delta=anchor_delta,
        dev_fold_deltas=dev_deltas,
        shadow_blend_delta=shadow_delta,
        shadow_blend_late_delta=shadow_late_delta,
        valid_anchor_metrics=m_v_a,
        valid_blend_metrics=m_v_b,
        plateau_span=plateau_span,
        test_anchor_corr=test_corr,
        seed_ensemble_stable=ens_stable,
    )
    record_id = dlog.pre_register(
        experiment_id="exp_012_framework_base",
        candidate_id="skeleton_anchor_recent_blend",
        params={"anchor_seg": [486, VALID_START], "recent_seg": [VALID_START - RECENT_LOOKBACK, VALID_START],
                "anchor_rounds": BASE_ROUNDS, "recent_rounds": RECENT_ROUNDS, "weight": selected_w,
                "seeds": list(DECISION_SEEDS), "note": "Train-only 口径；test_anchor_corr 门槛受训练终点不一致影响（见阶段1）"},
    )
    verdict = dlog.verify(record_id, gates_measured)
    log(f"DecisionLog: record={record_id} verdict={verdict['verdict']} "
        f"通过门槛数={sum(1 for g in verdict['outcomes'].values() if g['pass'])}/{len(verdict['outcomes'])}")

    summary = {
        "anchor_valid": ma["mean_rankic"], "recent_valid": mr["mean_rankic"],
        "selected_weight": selected_w, "shadow_delta": shadow_delta,
        "official_delta": m_v_b["mean_rankic"] - m_v_a["mean_rankic"],
        "test_anchor_corr": test_corr, "verdict": verdict["verdict"],
        "decision_record": record_id, "runtime_seconds": time.time() - started_all,
    }
    atomic = RESULT_DIR / "metrics.json.partial"
    atomic.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(atomic, RESULT_DIR / "metrics.json")
    log(f"骨架完成，总耗时 {time.time() - started_all:.1f}s。")
    return summary


if __name__ == "__main__":
    result = main()
    print("\n=== 阶段摘要 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
