"""exp_011 收尾：保存模型文件 + 可选低风险实验（分层抽样 / 小权重 TCN+线性融合 / 状态相似度）。

- 模型保存：model_anchor.* / model_recent.*（每个随机种子单独保存）
- 可选实验均为探索性，不改变主流程 prediction.npy。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
EXP_DIR = PROJECT_ROOT / "02_experiments" / "exp_011_stable_anchor_retrain"
sys.path.insert(0, str(EXP_DIR / "src"))

from dscr_exp011_lib import (  # noqa: E402
    BASE_ROUNDS, Dataset, RECENT_LOOKBACK, RECENT_ROUNDS, S, TEST_START, TEST_STOP,
    TRAIN_START, TRAIN_STOCK_CAP, VALID_START, VALID_STOP, WEIGHT_CANDIDATES,
    feature_cols_for_config, group_rank_transform, interval_target, json_ready,
    row_slice_for_times, score_prediction, train_ranker, predict_interval,
    stratified_capped_indices, build_training_arrays,
)
from dscr_exp011_lib import group_rank_ic_series

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
LOG_PATH = RESULT_DIR / "run.log"


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def atomic_write_text(path, text):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def main():
    ds = Dataset(DATASET_DIR, check_sha256=True)
    log("收尾脚本启动：保存模型 + 可选实验。")

    # =====================================================================
    # 1. 保存模型文件（锚点/近期专家 × 3 种子）
    # =====================================================================
    seg_anchor = [("train", TRAIN_START, VALID_START)]
    seg_recent = [("train", VALID_START - RECENT_LOOKBACK, VALID_START)]
    for mtype, segments, rounds in (("anchor", seg_anchor, BASE_ROUNDS), ("recent", seg_recent, RECENT_ROUNDS)):
        for seed in (42, 2026, 3407):
            cols = feature_cols_for_config("full_328")
            model, info = train_ranker(ds, segments, rounds, seed=seed, cols=cols)
            tag = f"model_{mtype}_seed{seed}"
            atomic_write_text(RESULT_DIR / f"{tag}.txt", model.model_to_string())
            meta = {
                "model_type": mtype, "seed": int(seed), "rounds": int(rounds),
                "feature_config": "full_328", "feature_count": int(cols.size),
                "segments": [[s, int(a), int(b)] for s, a, b in segments],
                "training_seconds": info["training_seconds"], "train_rows": info["train_rows"],
                "lightgbm_version": __import__("lightgbm").__version__,
            }
            atomic_write_text(RESULT_DIR / f"{tag}.metadata.json", json.dumps(json_ready(meta), ensure_ascii=False, indent=2))
            log(f"已保存 {tag}.txt (train_rows={info['train_rows']}, {info['training_seconds']:.1f}s)")

    # =====================================================================
    # 2. 可选实验A：分层股票抽样 vs 确定性抽样（开发折2 锚点，8轮，seed42）
    # =====================================================================
    log("可选实验A：分层抽样 vs 确定性抽样（fold_2 anchor r8 s42）。")
    import lightgbm as lgb
    params = {"objective": "lambdarank", "metric": "None", "learning_rate": 0.0228695,
              "num_leaves": 79, "min_data_in_leaf": 147, "feature_fraction": 0.80936,
              "bagging_fraction": 0.647764, "bagging_freq": 1, "lambda_l1": 2.35724,
              "lambda_l2": 0.238705, "max_bin": 127, "label_gain": list(range(64)),
              "lambdarank_truncation_level": 1024, "verbosity": -1, "seed": 42,
              "feature_fraction_seed": 42, "bagging_seed": 42, "num_threads": 30}
    strat_rows = []
    rng = np.random.default_rng(42)
    idx, capped = stratified_capped_indices(ds, "train", TRAIN_START, 2432, TRAIN_STOCK_CAP, rng)
    Xs = np.asarray(ds.tree["train"][idx, :328], dtype=np.float32)
    ys = np.asarray(ds.common["train"]["relevance"][idx], dtype=np.int8)
    gs = capped.astype(np.int32)
    dset = lgb.Dataset(Xs, label=ys, group=gs, free_raw_data=True)
    model_s = lgb.train(params, dset, num_boost_round=8, callbacks=[lgb.log_evaluation(0)])
    pred_s, gv = predict_interval(ds, model_s, "train", 2432, 2675, 8)
    y_fold = interval_target(ds, "train", 2432, 2675)
    m_s = score_prediction(pred_s, y_fold, gv)
    # 确定性抽样结果（fold_2 anchor，复用 driver 缓存键）
    from dscr_exp011_lib import model_fingerprint
    det_key_base = model_fingerprint("anchor_dev", [("train", TRAIN_START, 2432)], BASE_ROUNDS, 42, "full_328")
    det_cols = feature_cols_for_config("full_328")
    det_cols_hash = hashlib.sha256(np.ascontiguousarray(det_cols).tobytes()).hexdigest()[:8]
    det_path = RESULT_DIR / "runtime_cache" / f"{det_key_base}_{det_cols_hash}_train_2432_2675.npy"
    pred_d = np.load(det_path)
    m_d = score_prediction(pred_d, y_fold, gv)
    strat_rows.append({"comparison": "fold_2_anchor_r8_s42", "deterministic_rankic": m_d["mean_rankic"],
                       "stratified_rankic": m_s["mean_rankic"],
                       "delta": m_s["mean_rankic"] - m_d["mean_rankic"]})
    pd.DataFrame(strat_rows).to_csv(RESULT_DIR / "stratified_sampling.csv", index=False, encoding="utf-8-sig")
    log(f"分层抽样: det={m_d['mean_rankic']:.6f} strat={m_s['mean_rankic']:.6f} delta={m_s['mean_rankic'] - m_d['mean_rankic']:+.6f}")

    # =====================================================================
    # 3. 可选实验B：小权重 TCN + 线性 融合（官方 Valid 上检查；探索性）
    # =====================================================================
    log("可选实验B：小权重 TCN/线性 融合（官方 Valid，探索性）。")
    exp004_dir = PROJECT_ROOT / "04_results" / "exp_004_model_ensemble"
    comp_rows = []
    try:
        tcn_valid = np.load(exp004_dir / "component_valid_tcn.npy").astype(np.float32)
        lin_valid = np.load(exp004_dir / "component_valid_linear.npy").astype(np.float32)
        if tcn_valid.shape == (243, S) and lin_valid.shape == (243, S):
            valid_times = np.asarray(ds.common["valid"]["time"], dtype=np.int32)
            valid_stocks = np.asarray(ds.common["valid"]["stock"], dtype=np.int32)
            tcn_vec = tcn_valid[valid_times - VALID_START, valid_stocks]
            lin_vec = lin_valid[valid_times - VALID_START, valid_stocks]
            y_valid = interval_target(ds, "valid", VALID_START, VALID_STOP)
            groups_v = np.asarray(ds.common["valid"]["groups"], dtype=np.int32)
            # 锚点 = seed42 anchor valid raw prediction（driver 缓存）
            from dscr_exp011_lib import model_fingerprint, get_cached_prediction
            key = model_fingerprint("anchor", seg_anchor, BASE_ROUNDS, 42, "full_328")
            cached = get_cached_prediction(RESULT_DIR / "runtime_cache", f"{key}_valid_{VALID_START}_{VALID_STOP}", int(y_valid.size))
            anchor_vec = cached
            a_rank = group_rank_transform(anchor_vec, groups_v)
            t_rank = group_rank_transform(tcn_vec, groups_v)
            l_rank = group_rank_transform(lin_vec, groups_v)
            base_m = score_prediction(a_rank, y_valid, groups_v)
            for wa, wt, wl in ((0.85, 0.10, 0.05), (0.85, 0.15, 0.00), (0.90, 0.05, 0.05), (0.80, 0.15, 0.05)):
                blend = wa * a_rank + wt * t_rank + wl * l_rank
                mb = score_prediction(blend, y_valid, groups_v)
                comp_rows.append({"anchor_w": wa, "tcn_w": wt, "linear_w": wl,
                                  "valid_rankic": mb["mean_rankic"],
                                  "delta": mb["mean_rankic"] - base_m["mean_rankic"]})
                log(f"  融合 ({wa},{wt},{wl}): {mb['mean_rankic']:.6f} (Δ {mb['mean_rankic'] - base_m['mean_rankic']:+.6f})")
        else:
            log("  exp_004 分量网格 shape 异常，跳过。")
    except Exception as e:
        log(f"  小权重融合跳过（exp_004 分量不可用）：{e}")
    pd.DataFrame(comp_rows).to_csv(RESULT_DIR / "tiny_blend_results.csv", index=False, encoding="utf-8-sig")

    # =====================================================================
    # 4. 可选实验C：市场状态相似度加权（探索性）
    # =====================================================================
    log("可选实验C：市场状态相似度加权（探索性）。")
    try:
        # 时间级状态特征：每时点 池规模(mask_y数) 作为状态代理
        state_rows = []
        sample_times = np.arange(TRAIN_START, VALID_START, 8)
        groups_train = np.asarray(ds.common["train"]["groups"], dtype=np.int32)
        time_start_train = int(ds.common["train"]["time"][0])
        pool_size = {t: int(groups_train[t - time_start_train]) for t in sample_times}
        # 目标状态：官方 Valid 前段（2918-2968）的池规模均值
        groups_valid = np.asarray(ds.common["valid"]["groups"], dtype=np.int32)
        target_pool = float(np.mean(groups_valid[:50]))
        # 用 fold_2 锚点加权训练（full 权重 vs 状态相似度权重）
        from dscr_exp011_lib import capped_indices_for_split
        idx2, g2 = capped_indices_for_split(ds, "train", TRAIN_START, 2432, TRAIN_STOCK_CAP)
        X2 = np.asarray(ds.tree["train"][idx2, :328], dtype=np.float32)
        y2 = np.asarray(ds.common["train"]["relevance"][idx2], dtype=np.int8)
        times2 = np.asarray(ds.common["train"]["time"][idx2], dtype=np.int64)
        w_full = np.ones_like(times2, dtype=np.float32)
        w_state = np.ones_like(times2, dtype=np.float32)
        for t in np.unique(times2):
            m = times2 == t
            if t in pool_size:
                w_state[m] = float(np.exp(-abs(np.log(pool_size[t] / max(target_pool, 1))) / 2.0))
        w_state = (w_state / w_state.mean()).astype(np.float32)
        dset2 = lgb.Dataset(X2, label=y2, group=g2.astype(np.int32), weight=w_full, free_raw_data=True)
        model_f2 = lgb.train(params, dset2, num_boost_round=8, callbacks=[lgb.log_evaluation(0)])
        pred_f2, gv2 = predict_interval(ds, model_f2, "train", 2432, 2675, 8)
        m_f2 = score_prediction(pred_f2, y_fold, gv2)
        dset2s = lgb.Dataset(X2, label=y2, group=g2.astype(np.int32), weight=w_state, free_raw_data=True)
        model_f2s = lgb.train(params, dset2s, num_boost_round=8, callbacks=[lgb.log_evaluation(0)])
        pred_f2s, _ = predict_interval(ds, model_f2s, "train", 2432, 2675, 8)
        m_f2s = score_prediction(pred_f2s, y_fold, gv2)
        state_rows.append({
            "experiment": "fold_2_anchor_r8", "strategy": "full",
            "rankic": m_f2["mean_rankic"], "delta_vs_full": 0.0,
        })
        state_rows.append({
            "experiment": "fold_2_anchor_r8", "strategy": "state_similarity",
            "rankic": m_f2s["mean_rankic"], "delta_vs_full": m_f2s["mean_rankic"] - m_f2["mean_rankic"],
        })
        pd.DataFrame(state_rows).to_csv(RESULT_DIR / "state_similarity.csv", index=False, encoding="utf-8-sig")
        log(f"状态相似度: full={m_f2['mean_rankic']:.6f} state_w={m_f2s['mean_rankic']:.6f} delta={m_f2s['mean_rankic'] - m_f2['mean_rankic']:+.6f}")
    except Exception as e:
        log(f"状态相似度实验跳过：{e}")

    log("收尾脚本完成。")
    return {"models_saved": True, "optional_done": True}


if __name__ == "__main__":
    main()
