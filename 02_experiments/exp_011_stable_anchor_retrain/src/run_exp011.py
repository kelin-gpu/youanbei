"""exp_011 主流程驱动：锚点复现 -> 近期专家 -> 多随机种子 -> 三级验证 -> 权重选择
-> 重训模拟 -> 特征消融 -> 最终测试预测 -> 指标/元数据保存。

全部模型预测按指纹缓存到 <result_dir>/runtime_cache/，可安全重跑。
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
# 兼容：优先使用环境变量覆盖（Notebook 运行时可动态指定项目根目录）
if os.environ.get("DSCR_EXP011_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["DSCR_EXP011_PROJECT_ROOT"]).resolve()


def find_project_root():
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data.z").exists() and (candidate / "02_experiments").exists():
            return candidate
    raise RuntimeError("无法定位项目根目录：请从项目目录启动。")


if os.environ.get("DSCR_EXP011_AUTO_ROOT", "1") == "1":
    PROJECT_ROOT = find_project_root()

EXP_DIR = PROJECT_ROOT / "02_experiments" / "exp_011_stable_anchor_retrain"
sys.path.insert(0, str(EXP_DIR / "src"))

from dscr_exp011_lib import (  # noqa: E402
    ABLATION_CONFIGS, ANCHOR_EXPECTED_VALID_IC, ANCHOR_REPRO_TOLERANCE, BASE_ROUNDS,
    Dataset, RECENT_LOOKBACK, RECENT_ROUNDS, SEEDS, S, TEST_START, TEST_STOP,
    TEST_TIME_POINTS, TRAIN_START, VALID_START, VALID_STOP, WEIGHT_CANDIDATES,
    feature_cols_for_config, file_sha256, get_cached_prediction,
    group_rank_ic_series, group_rank_transform, interval_target, json_ready,
    model_fingerprint, predict_interval, rank_ic_self_test, row_slice_for_times,
    save_cached_prediction, score_prediction, train_ranker, validate_prediction_grid,
    vector_to_grid, mean_cross_sectional_rank_correlation,
)

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain"
CACHE_DIR = RESULT_DIR / "runtime_cache"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
ANCHOR_PATH = PROJECT_ROOT / "04_results" / "final_submission" / "prediction.npy"
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


def pred_key(model_type, segments, rounds, seed, config_name, split, start, stop, tag=""):
    base = model_fingerprint(model_type, segments, rounds, seed, config_name)
    cols = feature_cols_for_config(config_name)
    cols_hash = hashlib.sha256(np.ascontiguousarray(cols).tobytes()).hexdigest()[:8]
    return f"{base}_{cols_hash}_{tag}{split}_{int(start)}_{int(stop)}"


def ensure_prediction(ds, model_type, segments, rounds, seed, config_name, split, start, stop):
    """训练（如缓存缺失）并返回 (pred, groups, info_or_None)。"""
    key = pred_key(model_type, segments, rounds, seed, config_name, split, start, stop)
    rows, groups = row_slice_for_times(ds, split, start, stop)
    expected = int(rows.stop) - int(rows.start)
    cached = get_cached_prediction(CACHE_DIR, key, expected)
    if cached is not None:
        return cached, groups, None
    cols = feature_cols_for_config(config_name)
    model, info = train_ranker(ds, segments, rounds, seed=seed, cols=cols)
    pred, groups = predict_interval(ds, model, split, start, stop, rounds, cols=cols)
    save_cached_prediction(CACHE_DIR, key, pred)
    del model
    gc.collect()
    return pred, groups, info


def seed_ensemble(per_seed_preds, groups):
    ranks = [group_rank_transform(p, groups) for p in per_seed_preds]
    return np.mean(ranks, axis=0).astype(np.float32)


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
    log(f"exp_011 主流程启动。实验目录: {EXP_DIR}")
    log(f"结果目录: {RESULT_DIR} | 缓存目录: {CACHE_DIR}")

    ds = Dataset(DATASET_DIR, check_sha256=True)
    assert np.array_equal(feature_cols_for_config("full_328"), np.arange(328)), "full_328 列序必须为 [0:328)"
    log("数据契约校验通过（READY / manifest SHA-256 / legacy_328 兼容 / test_mask=2,042,538）。")
    log(f"RankIC 自检: {json.dumps(rank_ic_self_test(), ensure_ascii=False)}")

    # =====================================================================
    # 阶段 B：全历史锚点复现（官方 Valid）——锚点只训练到 2918（Train-only）
    # =====================================================================
    log("阶段B：训练全历史锚点 [486,2918) r8 s42，预测官方 Valid [2918,3161)。")
    seg_anchor_full = [("train", TRAIN_START, VALID_START)]
    pred_a_valid, groups_a, info_a = ensure_prediction(
        ds, "anchor", seg_anchor_full, BASE_ROUNDS, 42, "full_328", "valid", VALID_START, VALID_STOP)
    y_valid = interval_target(ds, "valid", VALID_START, VALID_STOP)
    metrics_a = score_prediction(pred_a_valid, y_valid, groups_a)
    repro_ok = abs(metrics_a["mean_rankic"] - ANCHOR_EXPECTED_VALID_IC) <= ANCHOR_REPRO_TOLERANCE
    log(f"锚点 Valid RankIC: {metrics_a['mean_rankic']:.6f} (期望 {ANCHOR_EXPECTED_VALID_IC:.6f}) 复现判定: {'通过' if repro_ok else '未通过'}")

    # =====================================================================
    # 阶段 C：近期 1702 期专家（官方 Valid + Test）
    # =====================================================================
    log("阶段C：训练近期专家 [1216,2918) r16 s42，预测 Valid 与 Test。")
    seg_recent_full = [("train", VALID_START - RECENT_LOOKBACK, VALID_START)]
    pred_r_valid, _, _ = ensure_prediction(
        ds, "recent", seg_recent_full, RECENT_ROUNDS, 42, "full_328", "valid", VALID_START, VALID_STOP)
    pred_r_test, groups_r_test, _ = ensure_prediction(
        ds, "recent", seg_recent_full, RECENT_ROUNDS, 42, "full_328", "test", TEST_START, TEST_STOP)
    metrics_r = score_prediction(pred_r_valid, y_valid, groups_a)
    rank_corr_ar = float(np.corrcoef(group_rank_transform(pred_a_valid, groups_a),
                                     group_rank_transform(pred_r_valid, groups_a))[0, 1])
    log(f"近期专家 Valid RankIC: {metrics_r['mean_rankic']:.6f}；与锚点秩相关: {rank_corr_ar:.4f}")

    # =====================================================================
    # 阶段 D：多随机种子（42/2026/3407）官方端点锚点与专家 -> Valid + Test
    # =====================================================================
    log("阶段D：多随机种子实验（42/2026/3407）。")
    anchor_seed_valid, anchor_seed_test = {}, {}
    recent_seed_valid, recent_seed_test = {}, {}
    seed_rows = []
    for seed in SEEDS:
        for mtype, segments, rounds in (
            ("anchor", seg_anchor_full, BASE_ROUNDS),
            ("recent", seg_recent_full, RECENT_ROUNDS),
        ):
            pv, gv, info = ensure_prediction(ds, mtype, segments, rounds, seed, "full_328", "valid", VALID_START, VALID_STOP)
            pt, gt, _ = ensure_prediction(ds, mtype, segments, rounds, seed, "full_328", "test", TEST_START, TEST_STOP)
            m = score_prediction(pv, y_valid, gv)
            seed_rows.append({
                "model_type": mtype, "seed": seed, "rounds": rounds,
                "full_rankic": m["mean_rankic"], "first_half_rankic": m["first_half_rankic"],
                "second_half_rankic": m["second_half_rankic"],
                "worst_quarter_rankic": m["worst_quarter_rankic"],
                "median_rankic": m["median_rankic"], "rankic_std": m["rankic_std"],
                "positive_ratio": m["positive_ratio"],
                "runtime": info["training_seconds"] if info else float("nan"),
                "prediction_hash": file_sha256(CACHE_DIR / f"{pred_key(mtype, segments, rounds, seed, 'full_328', 'valid', VALID_START, VALID_STOP)}.npy"),
            })
            if mtype == "anchor":
                anchor_seed_valid[seed] = pv
                anchor_seed_test[seed] = pt
            else:
                recent_seed_valid[seed] = pv
                recent_seed_test[seed] = pt
            log(f"  {mtype} seed={seed} Valid RankIC={m['mean_rankic']:.6f}")
    anchor_ens_valid = seed_ensemble([anchor_seed_valid[s] for s in SEEDS], groups_a)
    recent_ens_valid = seed_ensemble([recent_seed_valid[s] for s in SEEDS], groups_a)
    metrics_a_ens = score_prediction(anchor_ens_valid, y_valid, groups_a)
    metrics_r_ens = score_prediction(recent_ens_valid, y_valid, groups_a)
    log(f"锚点3种子集成 Valid RankIC: {metrics_a_ens['mean_rankic']:.6f} (std {metrics_a_ens['rankic_std']:.4f})")
    log(f"专家3种子集成 Valid RankIC: {metrics_r_ens['mean_rankic']:.6f} (std {metrics_r_ens['rankic_std']:.4f})")
    pd.DataFrame(seed_rows).to_csv(RESULT_DIR / "seed_results.csv", index=False, encoding="utf-8-sig")

    # =====================================================================
    # 阶段 E：开发折（折1 [2189,2432)、折2 [2432,2675)）
    # =====================================================================
    log("阶段E：开发折训练与权重搜索。")
    folds = [("fold_1", 2189, 2432), ("fold_2", 2432, 2675)]
    fold_preds = {}
    fold_rows = []
    for fold_name, valid_start, valid_stop in folds:
        seg_anchor = [("train", TRAIN_START, valid_start)]
        seg_recent = [("train", max(TRAIN_START, valid_start - RECENT_LOOKBACK), valid_start)]
        pa, gv, _ = ensure_prediction(ds, "anchor_dev", seg_anchor, BASE_ROUNDS, 42, "full_328", "train", valid_start, valid_stop)
        pr, _, _ = ensure_prediction(ds, "recent_dev", seg_recent, RECENT_ROUNDS, 42, "full_328", "train", valid_start, valid_stop)
        fold_preds[(fold_name, "anchor")] = pa
        fold_preds[(fold_name, "recent")] = pr
        y_fold = interval_target(ds, "train", valid_start, valid_stop)
        m_a = score_prediction(pa, y_fold, gv)
        m_r = score_prediction(pr, y_fold, gv)
        a_rank = group_rank_transform(pa, gv)
        r_rank = group_rank_transform(pr, gv)
        fold_rows.append({"fold": fold_name, "model": "anchor", "weight": 0.0, **m_a, "mean_delta": 0.0})
        fold_rows.append({"fold": fold_name, "model": "recent_expert", "weight": 0.0, **m_r,
                          "mean_delta": m_r["mean_rankic"] - m_a["mean_rankic"]})
        for w in WEIGHT_CANDIDATES:
            blend = (1 - w) * a_rank + w * r_rank
            mb = score_prediction(blend, y_fold, gv)
            fold_rows.append({"fold": fold_name, "model": "blend", "weight": w, **mb,
                              "mean_delta": mb["mean_rankic"] - m_a["mean_rankic"]})
        log(f"  {fold_name}: anchor={m_a['mean_rankic']:.6f} recent={m_r['mean_rankic']:.6f}")
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(RESULT_DIR / "fold_results.csv", index=False, encoding="utf-8-sig")

    weight_rows = []
    for w in WEIGHT_CANDIDATES:
        sub = fold_df[(fold_df["model"] == "blend") & (fold_df["weight"] == w)]
        d1 = float(sub[sub["fold"] == "fold_1"]["mean_delta"].iloc[0])
        d2 = float(sub[sub["fold"] == "fold_2"]["mean_delta"].iloc[0])
        m1 = float(sub[sub["fold"] == "fold_1"]["mean_rankic"].iloc[0])
        m2 = float(sub[sub["fold"] == "fold_2"]["mean_rankic"].iloc[0])
        weight_rows.append({
            "recent_weight": w, "fold_1_rankic": m1, "fold_2_rankic": m2,
            "fold_avg_rankic": (m1 + m2) / 2, "fold_1_delta": d1, "fold_2_delta": d2,
            "mean_delta": (d1 + d2) / 2, "worst_fold_delta": min(d1, d2),
        })
    wdf = pd.DataFrame(weight_rows)
    best_delta = float(wdf["mean_delta"].max())
    threshold = best_delta * 0.95
    near_best = wdf[wdf["mean_delta"] >= threshold].sort_values("recent_weight")
    selected_weight = float(near_best.iloc[0]["recent_weight"])
    wdf["selected"] = wdf["recent_weight"] == selected_weight
    wdf.to_csv(RESULT_DIR / "weight_search.csv", index=False, encoding="utf-8-sig")
    log(f"权重选择: 最佳平均增量={best_delta:.6f} -> 选定 recent_weight={selected_weight}")

    # =====================================================================
    # 阶段 F：影子验证 [2675,2918)（多种子 42/2026/3407）
    # =====================================================================
    log("阶段F：影子验证 [2675,2918) 多种子训练。")
    shadow_start, shadow_stop = 2675, 2918
    seg_anchor_sh = [("train", TRAIN_START, shadow_start)]
    seg_recent_sh = [("train", max(TRAIN_START, shadow_start - RECENT_LOOKBACK), shadow_start)]
    anchor_sh_preds, recent_sh_preds = {}, {}
    for seed in SEEDS:
        pa, g_sh, _ = ensure_prediction(ds, "anchor_shadow", seg_anchor_sh, BASE_ROUNDS, seed, "full_328", "train", shadow_start, shadow_stop)
        pr, _, _ = ensure_prediction(ds, "recent_shadow", seg_recent_sh, RECENT_ROUNDS, seed, "full_328", "train", shadow_start, shadow_stop)
        anchor_sh_preds[seed] = pa
        recent_sh_preds[seed] = pr
    y_shadow = interval_target(ds, "train", shadow_start, shadow_stop)
    anchor_sh_ens = seed_ensemble([anchor_sh_preds[s] for s in SEEDS], g_sh)
    recent_sh_ens = seed_ensemble([recent_sh_preds[s] for s in SEEDS], g_sh)
    m_sh_a_ens = score_prediction(anchor_sh_ens, y_shadow, g_sh)
    m_sh_r_ens = score_prediction(recent_sh_ens, y_shadow, g_sh)
    blend_sh = (1 - selected_weight) * anchor_sh_ens + selected_weight * recent_sh_ens
    m_sh_blend = score_prediction(blend_sh, y_shadow, g_sh)
    # 单种子 42 影子指标（多种子不稳定时的回退路径）
    m_sh_a_42 = score_prediction(anchor_sh_preds[42], y_shadow, g_sh)
    blend_sh_42 = ((1 - selected_weight) * group_rank_transform(anchor_sh_preds[42], g_sh)
                   + selected_weight * group_rank_transform(recent_sh_preds[42], g_sh))
    m_sh_blend_42 = score_prediction(blend_sh_42, y_shadow, g_sh)
    log(f"影子验证(3种子集成): anchor={m_sh_a_ens['mean_rankic']:.6f} recent={m_sh_r_ens['mean_rankic']:.6f} "
        f"blend(w={selected_weight})={m_sh_blend['mean_rankic']:.6f} 增量={m_sh_blend['mean_rankic'] - m_sh_a_ens['mean_rankic']:+.6f}")
    log(f"影子验证(单种子42): anchor={m_sh_a_42['mean_rankic']:.6f} blend={m_sh_blend_42['mean_rankic']:.6f} 增量={m_sh_blend_42['mean_rankic'] - m_sh_a_42['mean_rankic']:+.6f}")
    shadow_grid = vector_to_grid(group_rank_transform(blend_sh, g_sh), "train", ds, shadow_start, shadow_stop - shadow_start,
                                 interval=(shadow_start, shadow_stop))
    np.save(RESULT_DIR / "shadow_prediction.npy", shadow_grid.astype(np.float32))

    # =====================================================================
    # 阶段 G：官方 Valid 一次性检查（单种子与多种子集成）
    # =====================================================================
    log("阶段G：官方 Valid 一次性检查。")
    blend_use = (1 - selected_weight) * anchor_ens_valid + selected_weight * recent_ens_valid
    m_ov_anchor = metrics_a_ens
    m_ov_recent = metrics_r_ens
    m_ov_blend = score_prediction(blend_use, y_valid, groups_a)
    blend_42 = ((1 - selected_weight) * group_rank_transform(anchor_seed_valid[42], groups_a)
                + selected_weight * group_rank_transform(recent_seed_valid[42], groups_a))
    m_ov_blend_42 = score_prediction(blend_42, y_valid, groups_a)
    m_ov_anchor_42 = score_prediction(anchor_seed_valid[42], y_valid, groups_a)
    log(f"官方Valid(3种子集成): anchor={m_ov_anchor['mean_rankic']:.6f} recent={m_ov_recent['mean_rankic']:.6f} blend={m_ov_blend['mean_rankic']:.6f} 增量={m_ov_blend['mean_rankic'] - m_ov_anchor['mean_rankic']:+.6f}")
    log(f"官方Valid(单种子42): anchor={m_ov_anchor_42['mean_rankic']:.6f} blend={m_ov_blend_42['mean_rankic']:.6f}")
    valid_grid = vector_to_grid(group_rank_transform(blend_use, groups_a), "valid", ds, VALID_START, VALID_STOP - VALID_START)
    np.save(RESULT_DIR / "valid_prediction.npy", valid_grid.astype(np.float32))
    ens_stable = bool(
        m_ov_blend["mean_rankic"] >= m_ov_blend_42["mean_rankic"] - 0.0005
        and m_ov_blend["rankic_std"] <= m_ov_blend_42["rankic_std"] + 1e-9
        and m_sh_blend["mean_rankic"] >= 0
    )
    log(f"多种子集成稳定性判定: {ens_stable}")
    # 最终路径的官方 Valid 与影子指标（多种子不稳定 -> 单种子42；稳定 -> 3种子集成）
    if ens_stable:
        m_final_ov_anchor, m_final_ov_blend = m_ov_anchor, m_ov_blend
        m_final_sh_anchor, m_final_sh_blend = m_sh_a_ens, m_sh_blend
        official_delta_final = m_ov_blend["mean_rankic"] - m_ov_anchor["mean_rankic"]
        shadow_delta_final = m_sh_blend["mean_rankic"] - m_sh_a_ens["mean_rankic"]
    else:
        m_final_ov_anchor, m_final_ov_blend = m_ov_anchor_42, m_ov_blend_42
        m_final_sh_anchor, m_final_sh_blend = m_sh_a_42, m_sh_blend_42
        official_delta_final = m_ov_blend_42["mean_rankic"] - m_ov_anchor_42["mean_rankic"]
        shadow_delta_final = m_sh_blend_42["mean_rankic"] - m_sh_a_42["mean_rankic"]
    log(f"最终路径官方Valid: anchor={m_final_ov_anchor['mean_rankic']:.6f} blend={m_final_ov_blend['mean_rankic']:.6f} 增量={official_delta_final:+.6f}")
    log(f"最终路径影子验证: anchor={m_final_sh_anchor['mean_rankic']:.6f} blend={m_final_sh_blend['mean_rankic']:.6f} 增量={shadow_delta_final:+.6f}")

    # =====================================================================
    # 阶段 H：加入新标签重训模拟
    # =====================================================================
    log("阶段H：重训模拟。")
    # 模拟一：旧=训练至2189，新=训练至2432，预测 [2432,2675)
    s1_a, s1_b = 2432, 2675
    seg_anchor_f1 = [("train", TRAIN_START, 2189)]
    seg_recent_f1 = [("train", max(TRAIN_START, 2189 - RECENT_LOOKBACK), 2189)]
    old_a_s1, g_s1, _ = ensure_prediction(ds, "anchor_dev", seg_anchor_f1, BASE_ROUNDS, 42, "full_328", "train", s1_a, s1_b)
    old_r_s1, _, _ = ensure_prediction(ds, "recent_dev", seg_recent_f1, RECENT_ROUNDS, 42, "full_328", "train", s1_a, s1_b)
    new_a_s1, new_r_s1 = fold_preds[("fold_2", "anchor")], fold_preds[("fold_2", "recent")]
    y_s1 = interval_target(ds, "train", s1_a, s1_b)
    # 模拟二：旧=训练至2432（预测 [2675,2918)），新=训练至2675（预测 [2675,2918)）
    s2_a, s2_b = 2675, 2918
    old_a_s2, g_s2, _ = ensure_prediction(
        ds, "anchor_dev", [("train", TRAIN_START, 2432)], BASE_ROUNDS, 42, "full_328", "train", s2_a, s2_b)
    old_r_s2, _, _ = ensure_prediction(
        ds, "recent_dev", [("train", max(TRAIN_START, 2432 - RECENT_LOOKBACK), 2432)], RECENT_ROUNDS, 42, "full_328", "train", s2_a, s2_b)
    new_a_s2, new_r_s2 = anchor_sh_preds[42], recent_sh_preds[42]
    y_s2 = interval_target(ds, "train", s2_a, s2_b)
    g_s2 = g_sh

    sim_rows = []
    for sim, (y_tt, g_tt, old_p, new_p) in {
        "sim_1": (y_s1, g_s1, (old_a_s1, old_r_s1), (new_a_s1, new_r_s1)),
        "sim_2": (y_s2, g_s2, (old_a_s2, old_r_s2), (new_a_s2, new_r_s2)),
    }.items():
        old_blend = (1 - selected_weight) * group_rank_transform(old_p[0], g_tt) + selected_weight * group_rank_transform(old_p[1], g_tt)
        new_blend = (1 - selected_weight) * group_rank_transform(new_p[0], g_tt) + selected_weight * group_rank_transform(new_p[1], g_tt)
        old_m = score_prediction(old_blend, y_tt, g_tt)
        new_m = score_prediction(new_blend, y_tt, g_tt)
        old_ic = group_rank_ic_series(old_blend, y_tt, g_tt)
        new_ic = group_rank_ic_series(new_blend, y_tt, g_tt)
        diff = new_ic - old_ic
        sim_rows.append({
            "simulation": sim, "model": "old",
            "anchor_rankic": score_prediction(old_p[0], y_tt, g_tt)["mean_rankic"],
            "recent_rankic": score_prediction(old_p[1], y_tt, g_tt)["mean_rankic"],
            "blend_rankic": old_m["mean_rankic"],
            "first_half_rankic": old_m["first_half_rankic"],
            "second_half_rankic": old_m["second_half_rankic"],
            "worst_quarter_rankic": old_m["worst_quarter_rankic"],
        })
        sim_rows.append({
            "simulation": sim, "model": "new",
            "anchor_rankic": score_prediction(new_p[0], y_tt, g_tt)["mean_rankic"],
            "recent_rankic": score_prediction(new_p[1], y_tt, g_tt)["mean_rankic"],
            "blend_rankic": new_m["mean_rankic"],
            "first_half_rankic": new_m["first_half_rankic"],
            "second_half_rankic": new_m["second_half_rankic"],
            "worst_quarter_rankic": new_m["worst_quarter_rankic"],
        })
        sim_rows.append({
            "simulation": sim, "model": "delta",
            "blend_rankic": new_m["mean_rankic"] - old_m["mean_rankic"],
            "first_half_rankic": new_m["first_half_rankic"] - old_m["first_half_rankic"],
            "second_half_rankic": new_m["second_half_rankic"] - old_m["second_half_rankic"],
            "worst_quarter_rankic": new_m["worst_quarter_rankic"] - old_m["worst_quarter_rankic"],
            "per_time_diff_mean": float(np.nanmean(diff)),
            "per_time_diff_median": float(np.nanmedian(diff)),
            "per_time_diff_std": float(np.nanstd(diff)),
            "per_time_diff_min": float(np.nanmin(diff)),
            "per_time_diff_max": float(np.nanmax(diff)),
            "positive_time_share": float(np.nanmean(diff > 0)),
        })
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(RESULT_DIR / "retrain_simulation.csv", index=False, encoding="utf-8-sig")
    d1 = float(sim_df[(sim_df.simulation == "sim_1") & (sim_df.model == "delta")]["blend_rankic"].iloc[0])
    d2 = float(sim_df[(sim_df.simulation == "sim_2") & (sim_df.model == "delta")]["blend_rankic"].iloc[0])
    avg_delta = (d1 + d2) / 2
    retrain_pass = bool(avg_delta > 0 and d1 > -0.001 and d2 > -0.001)
    log(f"重训模拟: sim1 增量={d1:+.6f} sim2 增量={d2:+.6f} 平均={avg_delta:+.6f} -> {'通过' if retrain_pass else '未通过'}")

    # =====================================================================
    # 阶段 I：legacy_328 内部特征块消融（开发折、8轮、seed42）
    # =====================================================================
    log("阶段I：特征块消融。")
    ablation_rows = []
    base_by_fold = {}
    for fold_name, valid_start, valid_stop in folds:
        y_fold = interval_target(ds, "train", valid_start, valid_stop)
        _, gv = row_slice_for_times(ds, "train", valid_start, valid_stop)
        full_key = pred_key("anchor_dev", [("train", TRAIN_START, valid_start)], BASE_ROUNDS, 42, "full_328", "train", valid_start, valid_stop)
        full_pred = np.load(CACHE_DIR / f"{full_key}.npy")
        base_by_fold[fold_name] = score_prediction(full_pred, y_fold, gv)["mean_rankic"]
        for config_name in ABLATION_CONFIGS:
            if config_name == "full_328":
                continue
            t0 = time.time()
            pred, _, _ = ensure_prediction(ds, "anchor_abl", [("train", TRAIN_START, valid_start)],
                                           BASE_ROUNDS, 42, config_name, "train", valid_start, valid_stop)
            m = score_prediction(pred, y_fold, gv)
            ablation_rows.append({
                "feature_config": config_name,
                "feature_count": int(feature_cols_for_config(config_name).size),
                "fold": fold_name,
                "rankic": m["mean_rankic"],
                "delta_vs_full": m["mean_rankic"] - base_by_fold[fold_name],
                "runtime": time.time() - t0,
            })
            log(f"  消融 {config_name} {fold_name}: {m['mean_rankic']:.6f} (Δ {m['mean_rankic'] - base_by_fold[fold_name]:+.6f})")
    abl_df = pd.DataFrame(ablation_rows)
    summ_rows = []
    for config_name in [c for c in ABLATION_CONFIGS if c != "full_328"]:
        sub = abl_df[abl_df["feature_config"] == config_name]
        d1v = float(sub[sub["fold"] == "fold_1"]["delta_vs_full"].iloc[0]) if (sub["fold"] == "fold_1").any() else float("nan")
        d2v = float(sub[sub["fold"] == "fold_2"]["delta_vs_full"].iloc[0]) if (sub["fold"] == "fold_2").any() else float("nan")
        summ_rows.append({
            "feature_config": config_name,
            "feature_count": int(sub["feature_count"].iloc[0]),
            "fold_1_delta": d1v, "fold_2_delta": d2v,
            "mean_delta": (d1v + d2v) / 2,
            "worst_fold_delta": min(d1v, d2v),
            "total_runtime": float(sub["runtime"].sum()),
        })
    summ = pd.DataFrame(summ_rows)
    abl_out = abl_df.merge(summ, on=["feature_config", "feature_count"], suffixes=("", "_sum"))
    abl_out.to_csv(RESULT_DIR / "feature_ablation.csv", index=False, encoding="utf-8-sig")
    log(f"消融完成，共 {len(ablation_rows)} 次训练。")

    # =====================================================================
    # 阶段 J：最终测试预测
    # =====================================================================
    log("阶段J：最终测试预测生成。")
    test_mask = test_mask_of(ds)
    eval_count = int(test_mask.sum())

    # J1: Train-only（锚点/专家训练到2918）
    anchor_test_ens = seed_ensemble([anchor_seed_test[s] for s in SEEDS], groups_r_test)
    recent_test_ens = seed_ensemble([recent_seed_test[s] for s in SEEDS], groups_r_test)
    blend_test = (1 - selected_weight) * anchor_test_ens + selected_weight * recent_test_ens
    train_only_grid = vector_to_grid(group_rank_transform(blend_test, groups_r_test), "test", ds, TEST_START, TEST_TIME_POINTS).astype(np.float32)
    validate_prediction_grid(train_only_grid, test_mask)
    # 单种子 42 版本（多种子不稳定时的回退）
    blend_test_s42 = ((1 - selected_weight) * group_rank_transform(anchor_seed_test[42], groups_r_test)
                      + selected_weight * group_rank_transform(recent_seed_test[42], groups_r_test))
    train_only_grid_s42 = vector_to_grid(group_rank_transform(blend_test_s42, groups_r_test), "test", ds, TEST_START, TEST_TIME_POINTS).astype(np.float32)
    train_only_final_grid = train_only_grid_s42 if not ens_stable else train_only_grid
    validate_prediction_grid(train_only_final_grid, test_mask)
    np.save(RESULT_DIR / "train_only_prediction.npy", train_only_final_grid)

    # J2: Train+Valid 重训（锚点/专家训练到3161）
    seg_anchor_rv = [("train", TRAIN_START, VALID_START), ("valid", VALID_START, VALID_STOP)]
    seg_recent_rv = [("train", VALID_START - RECENT_LOOKBACK, VALID_START), ("valid", VALID_START, VALID_STOP)]
    anchor_rv_test, recent_rv_test = [], []
    for seed in SEEDS:
        pt, _, info = ensure_prediction(ds, "anchor_retrain", seg_anchor_rv, BASE_ROUNDS, seed, "full_328", "test", TEST_START, TEST_STOP)
        anchor_rv_test.append(pt)
        pt2, _, info2 = ensure_prediction(ds, "recent_retrain", seg_recent_rv, RECENT_ROUNDS, seed, "full_328", "test", TEST_START, TEST_STOP)
        recent_rv_test.append(pt2)
        a_str = "cached" if info is None else f"{info['training_seconds']:.1f}s"
        r_str = "cached" if info2 is None else f"{info2['training_seconds']:.1f}s"
        log(f"  重训 seed={seed} anchor {a_str} recent {r_str}")
    anchor_rv_ens = seed_ensemble(anchor_rv_test, groups_r_test)
    recent_rv_ens = seed_ensemble(recent_rv_test, groups_r_test)
    blend_rv = (1 - selected_weight) * anchor_rv_ens + selected_weight * recent_rv_ens
    retrain_grid = vector_to_grid(group_rank_transform(blend_rv, groups_r_test), "test", ds, TEST_START, TEST_TIME_POINTS).astype(np.float32)
    validate_prediction_grid(retrain_grid, test_mask)
    # 单种子 42 版本（多种子不稳定时的回退）
    blend_rv_s42 = ((1 - selected_weight) * group_rank_transform(anchor_rv_test[0], groups_r_test)
                    + selected_weight * group_rank_transform(recent_rv_test[0], groups_r_test))
    retrain_grid_s42 = vector_to_grid(group_rank_transform(blend_rv_s42, groups_r_test), "test", ds, TEST_START, TEST_TIME_POINTS).astype(np.float32)
    retrain_final_grid = retrain_grid_s42 if not ens_stable else retrain_grid
    validate_prediction_grid(retrain_final_grid, test_mask)
    np.save(RESULT_DIR / "train_valid_retrain_prediction.npy", retrain_final_grid)

    # J3: 最终选择（重训判定决定区间；多种子稳定性决定是否用集成）
    if retrain_pass:
        final_train_range, final_recent_range = "[486, 3161)", "[1459, 3161)"
        if ens_stable:
            final_grid, final_strategy = retrain_grid, "train_valid_retrain_seed_ensemble"
        else:
            final_grid, final_strategy = retrain_grid_s42, "train_valid_retrain_single_seed_42"
    else:
        final_train_range, final_recent_range = "[486, 2918)", "[1216, 2918)"
        if ens_stable:
            final_grid, final_strategy = train_only_grid, "train_only_anchor_recent_blend_seed_ensemble"
        else:
            final_grid, final_strategy = train_only_grid_s42, "train_only_anchor_recent_blend_single_seed_42"
    np.save(RESULT_DIR / "prediction.npy", final_grid)

    pred_check = validate_prediction_grid(final_grid, test_mask)
    anchor_grid = np.load(ANCHOR_PATH).astype(np.float32)
    test_corr = mean_cross_sectional_rank_correlation(final_grid, anchor_grid, groups_r_test,
                                                      np.asarray(ds.common["test"]["stock"], dtype=np.int32))
    pred_check["test_anchor_rank_correlation"] = test_corr
    pred_check["sha256"] = file_sha256(RESULT_DIR / "prediction.npy")
    log(f"最终预测: shape={pred_check['shape']} eval={pred_check['evaluation_count']} sha256={pred_check['sha256']} anchor_corr={test_corr:.4f}")

    # 晋级判定（基于最终选定路径：多种子稳定->集成，否则单种子42）
    official_delta = official_delta_final
    shadow_delta = shadow_delta_final
    dev_positive = bool((wdf["fold_1_delta"] >= 0).all() and (wdf["fold_2_delta"] >= 0).all())
    promoted = bool(
        best_delta >= 0.0005
        and dev_positive
        and shadow_delta > 0
        and official_delta > 0
        and (m_final_ov_blend["second_half_rankic"] >= m_final_ov_anchor["second_half_rankic"] - 0.0005)
        and (m_final_ov_blend["worst_quarter_rankic"] >= m_final_ov_anchor["worst_quarter_rankic"] - 0.0015)
        and test_corr >= 0.98
    )
    status = "promoted" if promoted else "not_promoted"
    log(f"晋级判定: {status}")

    metrics_payload = {
        "experiment_id": "exp_011_stable_anchor_retrain",
        "anchor_reproduction": {"expected": ANCHOR_EXPECTED_VALID_IC, "reproduced": metrics_a["mean_rankic"], "within_tolerance": repro_ok},
        "official_valid_anchor_s42": metrics_a,
        "official_valid_recent_s42": metrics_r,
        "official_valid_anchor_ens": m_ov_anchor,
        "official_valid_recent_ens": m_ov_recent,
        "official_valid_blend_ens": m_ov_blend,
        "official_valid_blend_s42": m_ov_blend_42,
        "official_delta_ens": official_delta_final,
        "shadow_anchor_ens": m_sh_a_ens,
        "shadow_recent_ens": m_sh_r_ens,
        "shadow_blend_ens": m_sh_blend,
        "shadow_delta_ens": m_sh_blend["mean_rankic"] - m_sh_a_ens["mean_rankic"],
        "final_path": {
            "official_valid_anchor": m_final_ov_anchor,
            "official_valid_blend": m_final_ov_blend,
            "official_delta": official_delta_final,
            "shadow_anchor": m_final_sh_anchor,
            "shadow_blend": m_final_sh_blend,
            "shadow_delta": shadow_delta_final,
        },
        "weight_search": wdf.to_dict(orient="records"),
        "selected_weight": selected_weight,
        "retrain_simulation": sim_df.to_dict(orient="records"),
        "retrain_deltas": {"sim_1": d1, "sim_2": d2, "average": avg_delta, "pass": retrain_pass},
        "seed_ensemble_stable": ens_stable,
        "promotion_status": status,
        "final_strategy": final_strategy,
        "final_prediction": pred_check,
        "runtime_seconds": time.time() - started_all,
    }
    atomic = RESULT_DIR / "metrics.json.partial"
    atomic.write_text(json.dumps(json_ready(metrics_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(atomic, RESULT_DIR / "metrics.json")

    metadata_payload = {
        "experiment_id": "exp_011_stable_anchor_retrain",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_hash": "a426a7078097e8d970c2f27a30a49b3122a8a0ea7c4c05f35938d5f568cfd04c",
        "cache_hash": "f7c4076de6e3ae7d631554df5a15f69f50d7e8f676249fb6d2d4cf71ccec8c6f",
        "feature_version": "legacy_328",
        "train_range": final_train_range,
        "recent_train_range": final_recent_range,
        "valid_range": "[2918, 3161)",
        "test_range": "[3161, 3603)",
        "model_parameters": {"lgb": "tuned_public_best(exp_003)", "learning_rate": 0.0228695, "num_leaves": 79, "min_data_in_leaf": 147, "feature_fraction": 0.80936, "bagging_fraction": 0.647764, "lambda_l1": 2.35724, "lambda_l2": 0.238705, "max_bin": 127, "objective": "lambdarank", "anchor_rounds": BASE_ROUNDS, "recent_rounds": RECENT_ROUNDS},
        "random_seeds": list(SEEDS),
        "selected_weight": selected_weight,
        "selected_training_strategy": final_strategy,
        "prediction_shape": pred_check["shape"],
        "prediction_dtype": "float32",
        "evaluation_position_count": eval_count,
        "non_evaluation_position_count": pred_check["non_evaluation_count"],
        "prediction_sha256": pred_check["sha256"],
        "promotion_status": status,
        "fallback_level": 0,
        "runtime_seconds": time.time() - started_all,
        "environment": "jingge_ts / Windows",
        "library_versions": {"numpy": np.__version__, "lightgbm": "4.7.0"},
        "formal_submission_overwritten": False,
    }
    atomic = RESULT_DIR / "metadata.json.partial"
    atomic.write_text(json.dumps(json_ready(metadata_payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(atomic, RESULT_DIR / "metadata.json")
    log(f"主流程完成，总耗时 {time.time()-started_all:.1f}s。")
    return {
        "anchor_valid": metrics_a["mean_rankic"], "recent_valid": metrics_r["mean_rankic"],
        "selected_weight": selected_weight, "shadow_blend": m_sh_blend["mean_rankic"],
        "official_blend": m_ov_blend["mean_rankic"], "retrain_pass": retrain_pass,
        "ens_stable": ens_stable, "status": status, "final_strategy": final_strategy,
        "prediction_sha256": pred_check["sha256"],
    }


if __name__ == "__main__":
    result = main()
    print("\n=== 阶段摘要 ===")
    for k, v in result.items():
        print(f"{k}: {v}")
