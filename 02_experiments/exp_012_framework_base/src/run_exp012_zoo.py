"""exp_012 阶段3：模型动物园。

候选动物（全部作为"近期 1702 期专家"与锚点融合，与 exp_009/010 骨架一致）：
  lgbm_xendcg     : LightGBM rank_xendcg 目标（同视图同标签）
  lgbm_cat337     : LightGBM lambdarank + legacy_328 + 9 类别列（类别感知 337 维）
  catboost_yetirank: CatBoost YetiRank（第二棵 GBDT）

流程（分阶段，缓存可断点续跑）：
  Stage A 筛选：fold_1/fold_2，seed42，与在任近期专家对比（相同权重网格）。
    通过条件：最佳平均增量 >= 0.0005 且两折均正，且不劣于在任专家（>= -0.0002）。
  Stage B 确认：通过者补 3 种子影子/官方 Valid/Test 全区间分量，跑完整晋级门槛。
  Stage C 融合：通过 Stage B 的动物进入受限融合（阶段4执行，本脚本只产出分量）。

所有选择指标不接触官方 Valid（仅在 Stage B 做一次性晋级检查）。
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
    RECENT_ROUNDS, S, TEST_START, TEST_STOP, TEST_TIME_POINTS, TRAIN_START,
    VALID_START, VALID_STOP,
    evaluate_gates, file_sha256, group_rank_transform, interval_target, json_ready,
    mean_cross_sectional_rank_correlation, row_slice_for_times, score_prediction,
    segment_anchor, segment_recent, seed_ensemble_preds, validate_prediction_grid,
    vector_to_grid,
)

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_012_model_zoo"
CACHE_DIR = RESULT_DIR / "runtime_cache"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
EXP011_CACHE = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain" / "runtime_cache"
DECISION_LOG_DIR = PROJECT_ROOT / "04_results" / "_decision_log"
LOG_PATH = RESULT_DIR / "run.log"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_GRID = (0.25, 0.30, 0.35)

ANIMALS = {
    "lgbm_xendcg": {"framework": "lgbm_xendcg", "cols": "legacy_328", "rounds": RECENT_ROUNDS},
    "lgbm_cat337": {"framework": "lgbm_lambdarank", "cols": "legacy_328_cats", "rounds": RECENT_ROUNDS},
    "catboost_yetirank": {"framework": "catboost", "cols": "legacy_328", "rounds": 120},
}

# 开发折定义（与 FOLD_SPECS 一致）
DEV_FOLDS = [("fold_1", 2189, 2432), ("fold_2", 2432, 2675)]
SHADOW = (2675, 2918)

log_lines = []


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_lines.append(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def animal_key(name):
    spec = ANIMALS[name]
    return {"model_type": f"zoo_{name}", "rounds": spec["rounds"], "cols": spec["cols"],
            "framework": spec["framework"]}


def blend_delta(anchor_raw, expert_raw, y, groups, w):
    blend = (1 - w) * group_rank_transform(anchor_raw, groups) + w * group_rank_transform(expert_raw, groups)
    return score_prediction(blend, y, groups)["mean_rankic"]


def main():
    started_all = time.time()
    open(LOG_PATH, "w", encoding="utf-8").close()
    cache = PredictionCache(CACHE_DIR, fallback_dirs=[EXP011_CACHE])
    dlog = DecisionLog(DECISION_LOG_DIR)
    ds = Dataset(DATASET_DIR, check_sha256=True)
    log(f"exp_012 model zoo 启动。结果目录: {RESULT_DIR}")

    # ------------------------------------------------------------------
    # Stage A：开发折筛选（seed 42）
    # ------------------------------------------------------------------
    screen_rows = []
    passed = {}
    for animal in ANIMALS:
        ak = animal_key(animal)
        log(f"[Stage A] 动物 {animal} ({ak['framework']}, {ak['cols']}, rounds={ak['rounds']})")
        best_mean_delta, best_w, fold_deltas = -1e9, None, []
        incumbent_deltas = []
        animal_fold_preds = {}
        for fold_name, v_start, v_stop in DEV_FOLDS:
            seg_a = segment_anchor(v_start)
            seg_r = segment_recent(v_start)
            pa, gv, _ = cache.get(ds, "anchor_dev", seg_a, BASE_ROUNDS, 42, "full_328",
                                  "train", v_start, v_stop)
            pr, _, _ = cache.get(ds, "recent_dev", seg_r, RECENT_ROUNDS, 42, "full_328",
                                 "train", v_start, v_stop)
            pa_an, _, _ = cache.get_animal(ds, ak["model_type"], seg_r, ak["rounds"], 42,
                                           ak["cols"], ak["framework"], "train", v_start, v_stop)
            animal_fold_preds[fold_name] = pa_an
            y = interval_target(ds, "train", v_start, v_stop)
            m_a = score_prediction(pa, y, gv)["mean_rankic"]
            d_inc = blend_delta(pa, pr, y, gv, 0.30) - m_a
            incumbent_deltas.append(d_inc)
            for w in WEIGHT_GRID:
                d = blend_delta(pa, pa_an, y, gv, w) - m_a
                screen_rows.append({"animal": animal, "fold": fold_name, "weight": w,
                                    "blend_delta": d, "incumbent_delta_030": d_inc})
            log(f"  {fold_name}: anchor={m_a:.6f} incumbent_Δ(w0.30)={d_inc:+.6f}")
        # 每动物：跨折最佳权重
        sub = pd.DataFrame(screen_rows)
        sub = sub[sub["animal"] == animal]
        for w in WEIGHT_GRID:
            ws = sub[sub["weight"] == w]
            d1 = float(ws[ws["fold"] == "fold_1"]["blend_delta"].iloc[0])
            d2 = float(ws[ws["fold"] == "fold_2"]["blend_delta"].iloc[0])
            mean_d = (d1 + d2) / 2
            if mean_d > best_mean_delta:
                best_mean_delta, best_w, fold_deltas = mean_d, w, [d1, d2]
        inc_best = float(np.mean(incumbent_deltas))
        ok = bool(best_mean_delta >= 0.0005 and all(d > 0 for d in fold_deltas)
                  and best_mean_delta >= inc_best - 0.0002)
        passed[animal] = ok
        log(f"  [筛选] 最佳Δ={best_mean_delta:+.6f} (w={best_w}) 两折={[f'{d:+.6f}' for d in fold_deltas]} "
            f"在任专家Δ={inc_best:+.6f} -> {'通过' if ok else '不通过'}")
        animal_fold_preds["anchor_mean"] = None
        if ok:
            passed[animal] = {"best_weight": best_w, "fold_deltas": fold_deltas,
                              "mean_delta": best_mean_delta, "incumbent_delta": inc_best,
                              "fold_preds": animal_fold_preds}
    screen_df = pd.DataFrame(screen_rows)
    screen_df.to_csv(RESULT_DIR / "zoo_screen.csv", index=False, encoding="utf-8-sig")

    # ------------------------------------------------------------------
    # Stage B：通过者 3 种子影子 / 官方 Valid / Test 全区间分量
    # ------------------------------------------------------------------
    zoo_rows = []
    for animal, status in passed.items():
        if not status:
            continue
        ak = animal_key(animal)
        best_w = status["best_weight"]
        log(f"[Stage B] 动物 {animal} 确认（w={best_w}）")
        # 影子 [2675,2918) 3 种子
        sh_start, sh_stop = SHADOW
        seg_r_sh = segment_recent(sh_start)
        seg_a_sh = segment_anchor(sh_start)
        a_sh, an_sh, r_sh = [], [], []
        for seed in DECISION_SEEDS:
            pa, g_sh, _ = cache.get(ds, "anchor_shadow", seg_a_sh, BASE_ROUNDS, seed, "full_328",
                                    "train", sh_start, sh_stop)
            pr, _, _ = cache.get(ds, "recent_shadow", seg_r_sh, RECENT_ROUNDS, seed, "full_328",
                                 "train", sh_start, sh_stop)
            pa_an, _, _ = cache.get_animal(ds, ak["model_type"], seg_r_sh, ak["rounds"], seed,
                                           ak["cols"], ak["framework"], "train", sh_start, sh_stop)
            a_sh.append(pa)
            r_sh.append(pr)
            an_sh.append(pa_an)
        y_sh = interval_target(ds, "train", sh_start, sh_stop)
        a_sh_ens = seed_ensemble_preds(a_sh, g_sh)
        an_sh_ens = seed_ensemble_preds(an_sh, g_sh)
        r_sh_ens = seed_ensemble_preds(r_sh, g_sh)
        m_sh_a = score_prediction(a_sh_ens, y_sh, g_sh)
        blend_an = (1 - best_w) * a_sh_ens + best_w * an_sh_ens
        blend_inc = (1 - 0.30) * a_sh_ens + 0.30 * r_sh_ens
        m_sh_an = score_prediction(blend_an, y_sh, g_sh)
        m_sh_inc = score_prediction(blend_inc, y_sh, g_sh)
        log(f"  影子: anchor={m_sh_a['mean_rankic']:.6f} 动物融合={m_sh_an['mean_rankic']:.6f} "
            f"Δ={m_sh_an['mean_rankic'] - m_sh_a['mean_rankic']:+.6f} "
            f"(在任 Δ={m_sh_inc['mean_rankic'] - m_sh_a['mean_rankic']:+.6f})")

        # 官方 Valid 一次性（3 种子）
        seg_a_v = segment_anchor(VALID_START)
        seg_r_v = segment_recent(VALID_START)
        a_v, an_v, r_v = [], [], []
        for seed in DECISION_SEEDS:
            pa, gv, _ = cache.get(ds, "anchor", seg_a_v, BASE_ROUNDS, seed, "full_328",
                                  "valid", VALID_START, VALID_STOP)
            pr, _, _ = cache.get(ds, "recent", seg_r_v, RECENT_ROUNDS, seed, "full_328",
                                 "valid", VALID_START, VALID_STOP)
            pa_an, _, _ = cache.get_animal(ds, ak["model_type"], seg_r_v, ak["rounds"], seed,
                                           ak["cols"], ak["framework"], "valid", VALID_START, VALID_STOP)
            a_v.append(pa)
            r_v.append(pr)
            an_v.append(pa_an)
        y_v = interval_target(ds, "valid", VALID_START, VALID_STOP)
        a_v_ens = seed_ensemble_preds(a_v, gv)
        an_v_ens = seed_ensemble_preds(an_v, gv)
        m_v_a = score_prediction(a_v_ens, y_v, gv)
        blend_v = (1 - best_w) * a_v_ens + best_w * an_v_ens
        m_v_an = score_prediction(blend_v, y_v, gv)
        log(f"  官方Valid: anchor={m_v_a['mean_rankic']:.6f} 动物融合={m_v_an['mean_rankic']:.6f} "
            f"Δ={m_v_an['mean_rankic'] - m_v_a['mean_rankic']:+.6f}")

        # Test 分量 + 锚点正交性
        seg_r_t = segment_recent(TEST_START)  # [1216,2918)
        an_t, r_t, a_t = [], [], []
        for seed in DECISION_SEEDS:
            pa_t, g_test, _ = cache.get(ds, "anchor", seg_a_v, BASE_ROUNDS, seed, "full_328",
                                        "test", TEST_START, TEST_STOP)
            pr_t, _, _ = cache.get(ds, "recent", seg_r_v, RECENT_ROUNDS, seed, "full_328",
                                   "test", TEST_START, TEST_STOP)
            pa_an, _, _ = cache.get_animal(ds, ak["model_type"], seg_r_v, ak["rounds"], seed,
                                           ak["cols"], ak["framework"], "test", TEST_START, TEST_STOP)
            a_t.append(pa_t)
            r_t.append(pr_t)
            an_t.append(pa_an)
        a_t_ens = seed_ensemble_preds(a_t, g_test)
        an_t_ens = seed_ensemble_preds(an_t, g_test)
        test_mask = np.zeros((TEST_TIME_POINTS, S), dtype=bool)
        test_mask[np.asarray(ds.common["test"]["time"], dtype=np.int32) - TEST_START,
                  np.asarray(ds.common["test"]["stock"], dtype=np.int32)] = True
        an_grid = vector_to_grid(group_rank_transform(an_t_ens, g_test), "test", ds,
                                 TEST_START, TEST_TIME_POINTS).astype(np.float32)
        validate_prediction_grid(an_grid, test_mask)
        anchor_grid = np.load(PROJECT_ROOT / "04_results" / "final_submission" / "prediction.npy").astype(np.float32)
        corr = mean_cross_sectional_rank_correlation(an_grid, anchor_grid, g_test,
                                                     np.asarray(ds.common["test"]["stock"], dtype=np.int32))
        log(f"  Test 分量已保存；与正式锚点截面秩相关={corr:.4f}")
        # 保存分量（向量形式，供阶段4融合）
        np.save(RESULT_DIR / f"animal_{animal}_shadow.npy", an_sh_ens)
        np.save(RESULT_DIR / f"animal_{animal}_valid.npy", an_v_ens)
        np.save(RESULT_DIR / f"animal_{animal}_test.npy", an_t_ens)
        zoo_rows.append({
            "animal": animal, "framework": ak["framework"], "cols": ak["cols"],
            "best_weight": best_w, "fold_deltas": status["fold_deltas"],
            "fold_mean_delta": status["mean_delta"], "incumbent_delta": status["incumbent_delta"],
            "shadow_delta": m_sh_an["mean_rankic"] - m_sh_a["mean_rankic"],
            "shadow_second_half_delta": m_sh_an["second_half_rankic"] - m_sh_a["second_half_rankic"],
            "official_valid_delta": m_v_an["mean_rankic"] - m_v_a["mean_rankic"],
            "official_valid_late_delta": m_v_an["second_half_rankic"] - m_v_a["second_half_rankic"],
            "official_valid_worst_quarter_delta": m_v_an["worst_quarter_rankic"] - m_v_a["worst_quarter_rankic"],
            "test_anchor_corr": corr,
        })
        dlog.pre_register(
            experiment_id="exp_012_model_zoo",
            candidate_id=f"animal_{animal}",
            params={"framework": ak["framework"], "cols": ak["cols"], "rounds": ak["rounds"],
                    "best_weight": best_w, "seeds": list(DECISION_SEEDS),
                    "note": "训练终点 Train-only(2918)；Test 分量锚点相关性门槛因官方锚点训练至 3161 而天然偏低"},
        )
    zoo_df = pd.DataFrame(zoo_rows)
    if not zoo_df.empty:
        zoo_df.to_csv(RESULT_DIR / "zoo_confirmation.csv", index=False, encoding="utf-8-sig")
    log(f"完成，总耗时 {time.time() - started_all:.1f}s。")
    return {"passed": [k for k, v in passed.items() if v],
            "rejected": [k for k, v in passed.items() if not v],
            "runtime_seconds": time.time() - started_all}


if __name__ == "__main__":
    result = main()
    print("\n=== 阶段摘要 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
