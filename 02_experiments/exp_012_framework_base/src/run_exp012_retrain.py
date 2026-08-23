"""exp_012 阶段1：训练终点策略决策实验。

问题：测试模型应训练到 2918（Train-only）还是 3161（含官方 Valid 标签）？
线上证据：exp_007/009 家族训练至 3161 -> 线上 0.109959/0.109928（最佳）；
exp_011 Train-only -> 线上 0.104988（低约 0.005）。本地 Valid 几乎相同。
exp_011 仅 2 个模拟（单种子），不足以定论，本实验扩展为 3 个模拟 x 3 种子。

模拟设计（old = 训练到 T0，new = 训练到 T1，比较两者对 [T1, T1+243) 的预测）：
  sim_A: 2189 -> 2432, 预测 [2432, 2675)
  sim_B: 2432 -> 2675, 预测 [2675, 2918)
  sim_C: 2675 -> 2918, 预测 [2918, 3161)  （官方 Valid，全部可评估）

每个模拟对锚点+近期 1702 期专家融合（w=0.30）比较，指标：均值/后半段/最差季度。
判定规则（与计划一致）：
  平均增量 > 0 且 任一模拟下探 > -0.001 且 后半段未恶化（>= -0.0005）-> 通过（统一训练至 3161）
  否则维持 Train-only（训练至 2918）并给出置信度。
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
    BASE_ROUNDS, DECISION_SEEDS, Dataset, DecisionLog, PredictionCache,
    RECENT_ROUNDS, TEST_START, TEST_STOP, VALID_START, VALID_STOP,
    file_sha256, group_rank_transform, interval_target, json_ready,
    row_slice_for_times, score_prediction, segment_anchor, segment_recent,
    seed_ensemble_preds, validate_prediction_grid, vector_to_grid,
)

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_012_retrain_policy"
CACHE_DIR = RESULT_DIR / "runtime_cache"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
EXP011_CACHE = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain" / "runtime_cache"
DECISION_LOG_DIR = PROJECT_ROOT / "04_results" / "_decision_log"
LOG_PATH = RESULT_DIR / "run.log"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT = 0.30  # 近期专家权重（exp_011 选定）

log_lines = []


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def blend_pair(anchor_raw, recent_raw, groups, weight=WEIGHT):
    return ((1 - weight) * group_rank_transform(anchor_raw, groups)
            + weight * group_rank_transform(recent_raw, groups))


def main():
    started_all = time.time()
    open(LOG_PATH, "w", encoding="utf-8").close()
    cache = PredictionCache(CACHE_DIR, fallback_dirs=[EXP011_CACHE])
    dlog = DecisionLog(DECISION_LOG_DIR)
    ds = Dataset(DATASET_DIR, check_sha256=True)
    log(f"exp_012 retrain policy 启动。结果目录: {RESULT_DIR}")

    # ------------------------------------------------------------------
    # 定义三个模拟的 (模型类型, 训练终点, 预测区间, 预测split)
    # ------------------------------------------------------------------
    sims = {
        "sim_A": {"old_end": 2189, "new_end": 2432, "pred": (2432, 2675), "split": "train"},
        "sim_B": {"old_end": 2432, "new_end": 2675, "pred": (2675, 2918), "split": "train"},
        "sim_C": {"old_end": 2675, "new_end": 2918, "pred": (VALID_START, VALID_STOP), "split": "valid"},
    }

    # 预注册决策
    record_id = dlog.pre_register(
        experiment_id="exp_012_retrain_policy",
        candidate_id="training_endpoint_2918_vs_3161",
        params={
            "simulations": [{"id": k, **v} for k, v in sims.items()],
            "blend_weight": WEIGHT,
            "anchor_rounds": BASE_ROUNDS, "recent_rounds": RECENT_ROUNDS,
            "seeds": list(DECISION_SEEDS),
            "decision_rule": "avg_mean_delta>0 and min_mean_delta>-0.001 and all_second_half_delta>=-0.0005",
        },
    )
    log(f"决策预注册: {record_id}")

    all_rows = []
    summary = {}
    for sim_name, spec in sims.items():
        pred_start, pred_stop = spec["pred"]
        split = spec["split"]
        y = interval_target(ds, split, pred_start, pred_stop)
        rows, groups = row_slice_for_times(ds, split, pred_start, pred_stop)
        g = groups
        log(f"--- {sim_name}: old_end={spec['old_end']} new_end={spec['new_end']} 预测 {split}[{pred_start},{pred_stop})")

        old_blends, new_blends = [], []
        for seed in DECISION_SEEDS:
            # old 模型（训练到 old_end）
            seg_a_o = segment_anchor(spec["old_end"])
            seg_r_o = segment_recent(spec["old_end"])
            pa_o, gv_o, _ = cache.get(ds, "anchor_dev", seg_a_o, BASE_ROUNDS, seed, "full_328",
                                      split, pred_start, pred_stop)
            pr_o, _, _ = cache.get(ds, "recent_dev", seg_r_o, RECENT_ROUNDS, seed, "full_328",
                                   split, pred_start, pred_stop)
            assert gv_o.size == g.size
            # new 模型（训练到 new_end）
            seg_a_n = segment_anchor(spec["new_end"])
            seg_r_n = segment_recent(spec["new_end"])
            pa_n, gv_n, _ = cache.get(ds, "anchor_dev", seg_a_n, BASE_ROUNDS, seed, "full_328",
                                      split, pred_start, pred_stop)
            pr_n, _, _ = cache.get(ds, "recent_dev", seg_r_n, RECENT_ROUNDS, seed, "full_328",
                                   split, pred_start, pred_stop)
            assert gv_n.size == g.size
            old_blends.append(blend_pair(pa_o, pr_o, g))
            new_blends.append(blend_pair(pa_n, pr_n, g))
            m_old = score_prediction(old_blends[-1], y, g)
            m_new = score_prediction(new_blends[-1], y, g)
            log(f"  seed={seed}: old={m_old['mean_rankic']:.6f} new={m_new['mean_rankic']:.6f} "
                f"Δ={m_new['mean_rankic'] - m_old['mean_rankic']:+.6f}")

        old_ens = seed_ensemble_preds(old_blends, g)
        new_ens = seed_ensemble_preds(new_blends, g)
        m_old_e = score_prediction(old_ens, y, g)
        m_new_e = score_prediction(new_ens, y, g)
        delta = m_new_e["mean_rankic"] - m_old_e["mean_rankic"]
        late_delta = m_new_e["second_half_rankic"] - m_old_e["second_half_rankic"]
        worstq_delta = m_new_e["worst_quarter_rankic"] - m_old_e["worst_quarter_rankic"]
        # 多种子 spread
        per_seed_deltas = []
        for i in range(len(DECISION_SEEDS)):
            m_o = score_prediction(old_blends[i], y, g)
            m_n = score_prediction(new_blends[i], y, g)
            per_seed_deltas.append(m_n["mean_rankic"] - m_o["mean_rankic"])
        summary[sim_name] = {
            "old_mean": m_old_e["mean_rankic"], "new_mean": m_new_e["mean_rankic"],
            "mean_delta": delta, "late_delta": late_delta, "worst_quarter_delta": worstq_delta,
            "per_seed_mean_deltas": per_seed_deltas,
            "seed_spread": float(np.max(per_seed_deltas) - np.min(per_seed_deltas)),
        }
        all_rows.append({
            "simulation": sim_name, "old_train_end": spec["old_end"], "new_train_end": spec["new_end"],
            "predict_range": f"[{pred_start},{pred_stop})", "old_blend_rankic": m_old_e["mean_rankic"],
            "new_blend_rankic": m_new_e["mean_rankic"], "mean_delta": delta,
            "second_half_delta": late_delta, "worst_quarter_delta": worstq_delta,
            "seed_deltas": str(per_seed_deltas),
        })
        log(f"  [集成] old={m_old_e['mean_rankic']:.6f} new={m_new_e['mean_rankic']:.6f} "
            f"Δmean={delta:+.6f} Δlate={late_delta:+.6f} Δworst_q={worstq_delta:+.6f}")

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------
    deltas = [summary[s]["mean_delta"] for s in sims]
    late_deltas = [summary[s]["late_delta"] for s in sims]
    avg_delta = float(np.mean(deltas))
    min_delta = float(np.min(deltas))
    late_ok = all(d >= -0.0005 for d in late_deltas)
    decision_pass = bool(avg_delta > 0 and min_delta > -0.001 and late_ok)
    confidence = "高" if (min_delta > 0 and all(d > 0.0002 for d in deltas)) else "中"
    verdict_text = ("通过：测试模型统一训练至 3161（含官方 Valid 标签）"
                    if decision_pass else "未通过：维持 Train-only（训练至 2918）")
    log(f"判定: 平均增量={avg_delta:+.6f} 最差模拟={min_delta:+.6f} 后半段全不恶化={late_ok} "
        f"-> {verdict_text}（置信度:{confidence}）")

    sim_df = pd.DataFrame(all_rows)
    sim_df.to_csv(RESULT_DIR / "retrain_simulation.csv", index=False, encoding="utf-8-sig")

    # 决策日志归档
    results = {
        "per_simulation": summary,
        "average_mean_delta": avg_delta,
        "min_mean_delta": min_delta,
        "all_second_half_not_worse": late_ok,
        "decision_pass": decision_pass,
        "verdict_text": verdict_text,
        "confidence": confidence,
    }
    verdict = dlog.verify(record_id, results)

    payload = {
        "experiment_id": "exp_012_retrain_policy",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_hash": "a426a7078097e8d970c2f27a30a49b3122a8a0ea7c4c05f35938d5f568cfd04c",
        "cache_hash": file_sha256(DATASET_DIR / "manifest.json"),
        "blend_weight": WEIGHT,
        "decision_record": record_id,
        "results": json_ready(results),
        "runtime_seconds": time.time() - started_all,
    }
    atomic = RESULT_DIR / "metrics.json.partial"
    atomic.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(atomic, RESULT_DIR / "metrics.json")
    log(f"完成，总耗时 {time.time() - started_all:.1f}s。")
    return {"decision_pass": decision_pass, "avg_delta": avg_delta, "min_delta": min_delta,
            "late_ok": late_ok, "confidence": confidence, "decision_record": record_id}


if __name__ == "__main__":
    result = main()
    print("\n=== 判定摘要 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
