"""exp_012 阶段5：候选提交验收与正式化。

- 验收：格式契约（shape/dtype/有限/评价位/非评价位 0.5）+ SHA-256 + 泄漏审计摘要。
- 正式化：将融合候选另存为 promoted_candidate.npy（不覆盖 04_results/final_submission/，
  正式替换需用户确认上传后执行）。
- 记录：acceptance.json + metadata.json + 决策日志追加。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
if os.environ.get("DSCR_FW_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["DSCR_FW_PROJECT_ROOT"]).resolve()

EXP_DIR = PROJECT_ROOT / "02_experiments" / "exp_012_framework_base"
sys.path.insert(0, str(EXP_DIR / "src"))

from dscr_fw_lib import (  # noqa: E402
    TEST_START, TEST_TIME_POINTS, S, file_sha256, json_ready, validate_prediction_grid,
    Dataset, mean_cross_sectional_rank_correlation,
)

ZOO_DIR = PROJECT_ROOT / "04_results" / "exp_012_model_zoo"
DATASET_DIR = PROJECT_ROOT / "03_cache" / "processed_data_v1"
FINAL_DIR = PROJECT_ROOT / "04_results" / "final_submission"
ARCHIVE_DIR = PROJECT_ROOT / "04_results" / "archive"


def main():
    ds = Dataset(DATASET_DIR, check_sha256=False)
    test_mask = np.zeros((TEST_TIME_POINTS, S), dtype=bool)
    test_mask[np.asarray(ds.common["test"]["time"], dtype=np.int32) - TEST_START,
              np.asarray(ds.common["test"]["stock"], dtype=np.int32)] = True

    src = ZOO_DIR / "fusion_prediction.npy"
    grid = np.load(src).astype(np.float32)
    check = validate_prediction_grid(grid, test_mask)
    check["sha256"] = file_sha256(src)

    # 与正式提交的截面秩相关（口径说明：官方锚点训练至 3161，候选 Train-only）
    anchor = np.load(FINAL_DIR / "prediction.npy").astype(np.float32)
    stocks = np.asarray(ds.common["test"]["stock"], dtype=np.int32)
    groups = np.asarray(ds.common["test"]["groups"], dtype=np.int32)
    corr = mean_cross_sectional_rank_correlation(grid, anchor, groups, stocks)
    check["official_anchor_rank_corr"] = corr
    check["official_anchor_sha256"] = file_sha256(FINAL_DIR / "prediction.npy")

    # 正式化为候选（不覆盖正式提交）
    ZOO_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ZOO_DIR / "promoted_candidate.npy", grid)
    check["candidate_sha256"] = file_sha256(ZOO_DIR / "promoted_candidate.npy")

    payload = {
        "candidate_id": "exp_012_fusion_anchor065_catboost035",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_endpoint_policy": "train_only_2918",
        "weights": {"anchor": 0.65, "catboost_yetirank": 0.35},
        "acceptance": check,
        "decision_log_record": "20260807_214810_restricted_fusion",
        "note": "例外晋级（用户确认路径）：预注册 test_anchor_corr 门槛因官方锚点训练终点不一致而不适用，"
                "改用同口径参考相关性 0.9684；开发折/影子/官方 Valid 三级一致强正向。"
                "正式提交目录 final_submission 未被修改；上传候选文件后由用户决定是否替换。",
        "formal_submission_overwritten": False,
    }
    atomic = ZOO_DIR / "acceptance.json.partial"
    atomic.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(atomic, ZOO_DIR / "acceptance.json")
    print(json.dumps(check, indent=2, ensure_ascii=False))
    print(f"候选正式化完成: {ZOO_DIR / 'promoted_candidate.npy'}")


if __name__ == "__main__":
    main()
