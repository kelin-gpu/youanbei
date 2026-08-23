"""保存 exp_011 线上成绩 0.104988 到 metrics.json / metadata.json / 报告。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

RESULT_DIR = Path(r"d:\google_dl\book\youanbei\04_results\exp_011_stable_anchor_retrain")
ONLINE_IC = 0.104988
SOURCE = "user_reported_2026-08-07"
NOW = time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_write_text(path, text):
    path = Path(path)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


# 1. metrics.json
mp = RESULT_DIR / "metrics.json"
m = json.loads(mp.read_text(encoding="utf-8"))
m["online_rank_ic"] = ONLINE_IC
m["online_result_source"] = SOURCE
m["online_recorded_at"] = NOW
m["online_vs_current_best"] = "below"  # 0.104988 < 0.109959 (exp_007)
atomic_write_text(mp, json.dumps(m, ensure_ascii=False, indent=2))
print("metrics.json 已更新：online_rank_ic =", ONLINE_IC)

# 2. metadata.json
dp = RESULT_DIR / "metadata.json"
d = json.loads(dp.read_text(encoding="utf-8"))
d["online_rank_ic"] = ONLINE_IC
d["online_result_source"] = SOURCE
d["online_recorded_at"] = NOW
atomic_write_text(dp, json.dumps(d, ensure_ascii=False, indent=2))
print("metadata.json 已更新。")

# 3. experiment_report.md 追加线上结果
erp = RESULT_DIR / "experiment_report.md"
er = erp.read_text(encoding="utf-8")
if "## 13. 线上结果" not in er:
    er = er.rstrip() + f"""

## 13. 线上结果

- 线上 RankIC：**{ONLINE_IC}**（来源：{SOURCE}，用户平台提交成绩）。
- 当前线上最佳为 `exp_007`（0.109959）；本次 `exp_011` 线上成绩未超过当前线上最佳。
- 说明：本地 Valid 融合 0.093634 与 exp_009（0.093615）基本持平，但线上 0.104988 低于 exp_003（0.108105）与 exp_007/exp_009（0.109959/0.109928），与本地排序不完全一致，体现本地 Valid 与线上的分布差异。
- 晋级结论不变：**not_promoted**；正式提交目录保持不变。
"""
    atomic_write_text(erp, er)
    print("experiment_report.md 已追加线上结果。")

# 4. prediction_report.md 更新
prp = RESULT_DIR / "prediction_report.md"
pr = prp.read_text(encoding="utf-8")
pr = pr.replace(
    "| exp_011（本次） | Train-only 锚点 0.70 + Train-only 近期专家 0.30（单种子 42） | 0.093634 | 未提交 |",
    f"| exp_011（本次） | Train-only 锚点 0.70 + Train-only 近期专家 0.30（单种子 42） | 0.093634 | {ONLINE_IC} |",
)
if "本次实验未超过当前已知线上最佳（exp_007 线上 RankIC 0.109959）" not in pr:
    pr = pr.replace(
        "- 本次实验未超过当前已知线上最佳（exp_007 线上 RankIC 0.109959），本地官方 Valid 融合 0.093634 与 exp_009（0.093615）基本持平。",
        f"- 本次实验线上 RankIC 为 **{ONLINE_IC}**（来源：{SOURCE}），未超过当前已知线上最佳（exp_007 线上 RankIC 0.109959）；本地官方 Valid 融合 0.093634 与 exp_009（0.093615）基本持平。",
    )
atomic_write_text(prp, pr)
print("prediction_report.md 已更新线上成绩。")
