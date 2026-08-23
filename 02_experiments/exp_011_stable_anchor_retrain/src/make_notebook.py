"""生成 exp_011 的独立可运行 Notebook（experiment.ipynb）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat as nbf

NB_PATH = Path(r"d:\google_dl\book\youanbei\02_experiments\exp_011_stable_anchor_retrain\experiment.ipynb")

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

md("""# exp_011 稳定锚点重训实验（Y1）

本 Notebook 独立完成：数据/缓存校验 → 锚点复现 → 近期 1702 期专家 → 多随机种子 → 三级验证（开发折/影子验证/官方 Valid）→ 权重选择 → 重训模拟 → 特征块消融 → 最终测试预测 → 指标与元数据保存。

- 特征：`legacy_328`（复用 `03_cache/processed_data_v1`，因果性已由缓存清单校验）。
- 模型：LightGBM LambdaRank（exp_003 调优参数）；全历史锚点 8 轮，近期专家 16 轮。
- 分组：每个时间点为 ranking group；每时点确定性抽取至多 1200 只股票。
- 结果：全部写入 `04_results/exp_011_stable_anchor_retrain/`，**不会覆盖** `04_results/final_submission/prediction.npy`。
- 所有模型预测按指纹缓存到结果目录 `runtime_cache/`，重复运行自动复用。

内核建议：`jingge_ts`（python 3.10 + lightgbm 4.7.0）。""")

md("""## 1. 环境与路径

从当前目录向上定位项目根目录，并切换到项目根目录。""")

code_setup = """from __future__ import annotations
import os, sys
from pathlib import Path

def find_project_root():
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data.z").exists() and (candidate / "02_experiments").exists():
            return candidate
    raise RuntimeError("无法定位项目根目录：请从项目目录或实验目录启动 Notebook。")

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
os.environ["DSCR_EXP011_AUTO_ROOT"] = "1"
print("项目根目录:", PROJECT_ROOT)"""

cells.append(nbf.v4.new_code_cell(code_setup))

md("""## 2. 数据契约与 RankIC 自检

`Dataset` 会校验 READY、manifest SHA-256、legacy_328 兼容性与测试评价位置数（2,042,538）。`rank_ic_self_test` 在手工构造样本上验证官方 RankIC 实现（正/负相关、与 scipy 对照、常量、NaN/Inf、并列、样本过少）。""")

code_contract = """import sys
sys.path.insert(0, str(PROJECT_ROOT / "02_experiments" / "exp_011_stable_anchor_retrain" / "src"))

from dscr_exp011_lib import Dataset, rank_ic_self_test, feature_cols_for_config
import numpy as np

assert list(feature_cols_for_config("full_328")) == list(range(328)), "full_328 列序必须为 [0:328)"
print("RankIC 自检:", rank_ic_self_test())

ds = Dataset(PROJECT_ROOT / "03_cache" / "processed_data_v1", check_sha256=True)
print(ds.split_meta().to_string(index=False))
print("数据契约校验通过。")"""

cells.append(nbf.v4.new_code_cell(code_contract))

md("""## 3. 完整实验流水线

执行锚点复现（目标 Valid RankIC ≈ 0.092940）、近期专家、多随机种子（42/2026/3407）、开发折权重搜索（0.25/0.30/0.35，选定 95% 阈值内最低权重）、影子验证 [2675,2918)、官方 Valid 一次性检查、重训模拟（模拟一/二）、legacy_328 内部特征块消融、最终测试预测（Train-only 与 Train+Valid 重训两个版本）与 `prediction.npy` 生成。

所有阶段均有指纹缓存，中断后重新运行会自动续跑。""")

code_pipeline = """from run_exp011 import main as run_exp011_main

result = run_exp011_main()
print("\\n=== 阶段摘要 ===")
for k, v in result.items():
    print(f"{k}: {v}")"""

cells.append(nbf.v4.new_code_cell(code_pipeline))

md("""## 4. 最终预测文件核验

读回 `prediction.npy`，按官方口径复核 shape、dtype、有限性、评价/非评价位置数与填充值，并输出 SHA-256。""")

code_verify = """import hashlib, json
import numpy as np

RESULT_DIR = PROJECT_ROOT / "04_results" / "exp_011_stable_anchor_retrain"

def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(16 * 1024 * 1024):
            d.update(c)
    return d.hexdigest()

loaded = np.load(RESULT_DIR / "prediction.npy")
test_mask = np.zeros((442, 5282), dtype=bool)
test_mask[np.asarray(ds.common["test"]["time"], dtype=np.int32) - 3161,
          np.asarray(ds.common["test"]["stock"], dtype=np.int32)] = True

summary = {
    "prediction_path": str(RESULT_DIR / "prediction.npy"),
    "shape": list(loaded.shape),
    "dtype": str(loaded.dtype),
    "finite": bool(np.isfinite(loaded).all()),
    "evaluation_count": int(test_mask.sum()),
    "non_evaluation_count": int((~test_mask).sum()),
    "non_evaluation_all_0_5": bool(np.all(loaded[~test_mask] == 0.5)),
    "min": float(loaded.min()), "max": float(loaded.max()),
    "mean": float(loaded.mean()), "std": float(loaded.std()),
    "sha256": sha256(RESULT_DIR / "prediction.npy"),
}
for k, v in summary.items():
    print(f"{k}: {v}")

assert summary["shape"] == [442, 5282] and summary["dtype"] == "float32"
assert summary["finite"] and summary["evaluation_count"] == 2_042_538 and summary["non_evaluation_all_0_5"]

metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
print("\\n最终策略:", metrics["final_strategy"])
print("晋级状态:", metrics["promotion_status"])
print("正式提交目录是否被修改: False（本实验不写 final_submission）")
print("\\nexp_011 实验完成：prediction.npy 与 prediction_report.md 均已生成。")"""

cells.append(nbf.v4.new_code_cell(code_verify))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (jingge_ts)", "language": "python", "name": "jingge_ts"},
    "language_info": {"name": "python", "version": "3.10.20"},
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("已生成:", NB_PATH)
