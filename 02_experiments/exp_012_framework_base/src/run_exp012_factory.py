"""exp_012 阶段2：特征工厂准入裁决。

从 block_registry.json + 历史证据 + 转移衰减表生成逐块准入裁决表。
新增块（Phase 3 产生新证据后）通过 adjudicate_block 复用同一门槛。

准入门槛（与计划一致）：
  fold_internal    : 3 开发折平均增量 >= 0.0010 且折1/2均正且后段>=0（种子集成口径）
  transfer_budget  : 同族历史衰减率 >= 0.5，或折内增量 >= 2x0.0005
  seed_consistency : 多种子增益符号一致（spread <= 0.002）
  orthogonality    : 候选专家与锚点 Test 截面秩相关 < 0.99
  placebo          : 时间重排后无增益（防泄漏，一票否决）
  imputation       : 补全质量在 Train/Valid/Test 间漂移在阈值内（exp_008 家族）
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"d:\google_dl\book\youanbei")
if os.environ.get("DSCR_FW_PROJECT_ROOT"):
    PROJECT_ROOT = Path(os.environ["DSCR_FW_PROJECT_ROOT"]).resolve()

EXP_DIR = PROJECT_ROOT / "02_experiments" / "exp_012_framework_base"
sys.path.insert(0, str(EXP_DIR / "src"))

from dscr_fw_lib import TRANSFER_TABLE  # noqa: E402

FACTORY_DIR = PROJECT_ROOT / "04_results" / "_feature_factory"
REGISTRY_PATH = FACTORY_DIR / "block_registry.json"
OUT_CSV = FACTORY_DIR / "admission_adjudication.csv"

# 历史证据（来自 exp_005/008/010/011，作为裁决输入；Phase 3 新证据会覆盖 pending 项）
HISTORICAL_EVIDENCE = {
    "numeric_408_diff": {
        "fold_internal": "历史(exp_005): 408 相对 328 无可靠提升 -> 不通过",
        "note": "exp_005 以整体视图筛选，未做块级多种子；可在 Phase 3 复核",
    },
    "tree_419_categorical": {
        "fold_internal": "通过（exp_012 zoo: cat337 两折均正，最佳Δ+0.001307 >= 0.0010 且优于在任专家）",
        "transfer_budget": "通过（exp_012 zoo Stage B: 官方 Valid Δ+0.000754 > 0；影子 +0.000719）",
        "seed_consistency": "待补充（Stage B 使用 3 种子集成，折级为 seed42）",
        "orthogonality": "通过（Test 与锚点秩相关 0.8663 < 0.99）",
        "placebo": "待执行（时间重排安慰剂）",
        "note": "exp_012 在专家+门槛语境首次单独测过类别块并确认增益；此为项目内首个通过折内+Valid 双门槛的特征块",
    },
    "exp008_surprise": {
        "fold_internal": "通过（exp_010: 3 折均正, 增量+0.000890 >= 0.0005 但 < 0.0010 严格门槛）",
        "transfer_budget": "不通过（官方 Valid w=0.00，衰减率~0）",
        "note": "折内未达 0.0010 且迁移衰减为 0，双门槛均不满足",
    },
    "exp008_trend": {"transfer_budget": "不通过（exp_008 全体系 Valid -0.0085）"},
    "exp008_history_state": {"transfer_budget": "不通过（随 exp_008 体系）"},
    "exp008_raw99": {"note": "仅作为 363 视图组成部分，不单独评估"},
    "exp008_cs_rank99": {"note": "仅作为 363 视图组成部分，不单独评估"},
    "exp008_surprise_rank": {"note": "从未单独评估 -> 待裁决"},
}


def decay_rate_of(candidate: str) -> float | None:
    for row in TRANSFER_TABLE:
        if row["candidate"] == candidate:
            return row["decay_rate"]
    return None


def main():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = []
    for block_id, block in registry["blocks"].items():
        evidence = HISTORICAL_EVIDENCE.get(block_id, {})
        decay = decay_rate_of(block_id)
        row = {
            "block_id": block_id,
            "view": block["view"],
            "feature_count": block["feature_count"],
            "column_ranges": str(block["column_ranges"]),
            "source_experiment": block["source_experiment"],
            "fold_internal": evidence.get("fold_internal", "待裁决(Phase 3)"),
            "transfer_budget": evidence.get("transfer_budget", f"历史衰减率={decay if decay is not None else 'N/A'}"),
            "seed_consistency": evidence.get("seed_consistency", "待裁决"),
            "orthogonality": evidence.get("orthogonality", "待裁决"),
            "placebo": evidence.get("placebo", "未执行（无增益候选时可不执行）"),
            "imputation_quality": evidence.get("imputation_quality", "N/A（非补全族）" if "exp008" not in block_id else "待裁决"),
            "note": evidence.get("note", ""),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    FACTORY_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(frame.to_string(index=False))
    print(f"\n准入裁决表已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
