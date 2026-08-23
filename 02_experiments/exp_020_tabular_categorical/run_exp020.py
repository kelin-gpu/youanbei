from __future__ import annotations

"""exp_020：cat_5 原生类别纳入 tabular 专家（重训 tabular + 复用其它家族重新融合）。

对 exp016 的 tabular 家族做单变量改动：LightGBM 的 lgbm_rank / lgbm_huber 显式
把 9 个类别列（含 cat_5，列 408..416）作为原生类别特征（categorical_feature），
与 CatBoost 对齐；其余神经网络家族、元头、路由权重全部复用 exp016 既有产物。

产出：重训后的 tabular 家族 + 重新融合的 Test 提交，以及全量 Valid 上的本地参照指标。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, r"D:\google_dl\book\youanbei\02_experiments")

from exp_016_unified_expert_fusion.config import (FAMILIES, RunConfig, STOCK_COUNT,
                                                  TEST_START, TEST_STOP, VALID_START, VALID_STOP)
from exp_016_unified_expert_fusion.src.data_context import DataContext
from exp_016_unified_expert_fusion.src.full_pipeline import _combined_supervised_arrays, _tabular_arrays
from exp_016_unified_expert_fusion.src.prediction_contract import validate_prediction, vector_to_grid
from exp_016_unified_expert_fusion.src.ranking import dynamic_blend_family_predictions, rank_ic
from exp_016_unified_expert_fusion.src.tabular_experts import predict_tabular_family, train_tabular_family

EXP016_FULL = Path(r"D:\google_dl\book\youanbei\04_results\exp_016_unified_expert_fusion\full")
RESULT = Path(r"D:\google_dl\book\youanbei\04_results\exp_020_tabular_categorical")


def per_group_ic(pred, target, groups):
    scores, offset = [], 0
    for size in groups:
        size = int(size)
        scores.append(rank_ic(pred[offset:offset + size], target[offset:offset + size]))
        offset += size
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite))


def main():
    t0 = time.time()
    RESULT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = RunConfig(mode="full", training_allowed=True, stage="all", device=device, stock_cap=1024)
    ctx = DataContext(load_sequence=False)

    # 1) 重训最终 tabular：[486, 3161) capped 1024，预测 Test
    final_X, final_y, final_rel, final_groups = _combined_supervised_arrays(ctx, config.stock_cap)
    print(f"[exp020] final tabular train rows={final_X.shape[0]}", flush=True)
    models = train_tabular_family(config, final_X, final_y, final_rel, final_groups)
    del final_X, final_y, final_rel, final_groups

    test_X, _, _, test_groups, _ = _tabular_arrays(ctx, "test", TEST_START, TEST_STOP, STOCK_COUNT)
    tabular_test, _ = predict_tabular_family(models, test_X, test_groups)
    del test_X

    # 2) 复用 exp016 其它家族 + 路由权重，重新融合
    family = {}
    for name in FAMILIES:
        family[name] = tabular_test.astype(np.float32) if name == "tabular" else np.load(EXP016_FULL / f"family_{name}.npy").astype(np.float32)
    weights = np.load(EXP016_FULL / "dynamic_weights.npy")  # (442, 7)
    blended = dynamic_blend_family_predictions(family, test_groups, weights)
    grid = vector_to_grid(blended, ctx.common["test"]["time"], ctx.common["test"]["stock"], TEST_START, TEST_STOP - TEST_START)
    contract = validate_prediction(grid, ctx.test_evaluation_mask())
    np.save(RESULT / "prediction_1.npy", grid)
    print(f"[exp020] test submission contract={contract}", flush=True)

    # 3) 本地参照：Train-only [486,2918) capped 1024 重训 tabular，预测全量 Valid
    train_X, train_y, train_rel, train_groups, _ = _tabular_arrays(ctx, "train", 486, VALID_START, config.stock_cap)
    valid_X, valid_y, _, valid_groups, _ = _tabular_arrays(ctx, "valid", VALID_START, VALID_STOP, STOCK_COUNT)
    models_val = train_tabular_family(config, train_X, train_y, train_rel, train_groups)
    del train_X, train_y, train_rel, train_groups
    tabular_valid, _ = predict_tabular_family(models_val, valid_X, valid_groups)
    del valid_X

    tabular_ic = per_group_ic(tabular_valid, valid_y, valid_groups)
    print(f"[exp020] new tabular standalone full-valid IC = {tabular_ic:.6f}", flush=True)

    # 用 T0.1b 的全量 Valid 七家族预测 + 权重，替换 tabular 后重融合
    full_family = np.load(EXP016_FULL / "full_valid_family_predictions.npy").copy()
    full_w = np.load(EXP016_FULL / "full_valid_dynamic_weights.npy")
    full_family[:, 1] = tabular_valid
    valid_family = {name: full_family[:, i] for i, name in enumerate(FAMILIES)}
    reblend_ic = per_group_ic(dynamic_blend_family_predictions(valid_family, valid_groups, full_w), valid_y, valid_groups)
    print(f"[exp020] re-blended full-valid IC = {reblend_ic:.6f} (baseline router 0.091389)", flush=True)

    metrics = {
        "experiment": "exp_020_tabular_categorical",
        "change": "tabular 家族 LightGBM 将 9 个类别列(408..416，含 cat_5)作为原生类别特征",
        "new_tabular_standalone_full_valid_ic": tabular_ic,
        "reblend_full_valid_ic": reblend_ic,
        "baseline_router_full_valid_ic": 0.091389,
        "reblend_delta_vs_router": round(reblend_ic - 0.091389, 6),
        "contract": contract,
        "submission": "prediction_1.npy",
    }
    metadata = {
        "experiment": "exp_020_tabular_categorical",
        "task": "cat_5 native categorical into tabular family",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_submission_overwritten": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (RESULT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp020] DONE in {time.time() - t0:.1f}s", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
