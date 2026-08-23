# exp_009_anchor_recent_blend_cv

状态：`historical`（历史候选，不作为当前主基线）。

入口：`experiment.ipynb`

该实验固定使用 `legacy_328` 和正式提交作为锚点，在 Train 内三折走步验证中选择最近 1702 期专家的 8/16 轮配置与 `0%~35%` 融合权重。

默认直接在 `jingge_ts` 内核中从上到下运行。完整结果写入：

```text
04_results/exp_009_anchor_recent_blend_cv/
```

无论是否通过晋级门槛，完整运行都会生成 `prediction.npy`、`expert_prediction.npy`、`valid_prediction.npy`、指标、元数据、CSV 明细和中文实验报告。实验不会覆盖 `04_results/final_submission/prediction.npy`。

如只需检查环境和输出契约，可在启动 Jupyter 前设置：

```powershell
$env:DSCR_EXP009_MODE = "preflight"
```

正式训练使用默认 `full` 模式。训练会读取约 34.8 GiB 的共享缓存并多次训练 LightGBM，请预留足够内存和时间。
