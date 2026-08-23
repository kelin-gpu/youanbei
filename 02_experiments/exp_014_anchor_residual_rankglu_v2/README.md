# exp_014 v2 运行说明

状态：`historical_rejected`（本地提升未迁移到线上）。

本实验使用 `jingge_ts` Jupyter 内核（已验证：Python 3.10.20、PyTorch 2.6.0+cu124、CUDA）。`build_notebook.py` 是唯一源码，`experiment.ipynb` 是生成物；修改逻辑后必须重新运行生成脚本。

## 正式运行

1. 用 `jingge_ts` 内核打开 `experiment.ipynb`。
2. 确认没有设置 `DSCR_EXP014_V2_MODE`，或将其设为 `full`。
3. 执行 **Restart Kernel and Run All Cells**。

正式训练最多 25 个 epoch。RTX 4060 等支持 BF16 的 CUDA 设备优先使用 BF16；数值异常的截面会自动用 FP32 重试一次。正式结果写入：

`04_results/exp_014_anchor_residual_rankglu_v2/`

完成标志是 `metadata.json` 中的 `status: "full_completed"`。推荐提交文件始终为 `prediction.npy`：候选通过全部门槛时它是候选预测，否则自动回退为 anchor。`submission_choice.json` 会明确记录选择结果。

## 中断恢复与故障定位

新版断点写入 `models_v4/<stage>/seed_<seed>/<fingerprint>/`，每个完整 epoch 后原子更新。重新 Run All 会从最后一个完整 epoch 继续；旧版 `models/` 文件保留但不会加载。损坏或不兼容的新版断点会被重命名为 `.rejected_*` 后从该阶段干净重训。

如果 BF16/FP16 与 FP32 均出现非有限数值，训练会停止并把 stage、seed、epoch、fold、time 和各次精度尝试写入 `failures/*.json`。

## 非正式检查

在启动 Jupyter 前设置：

```powershell
$env:DSCR_EXP014_V2_MODE = "smoke"
```

`smoke` 只使用合成张量检查数值稳定性、FP32 重试、checkpoint 和完整控制流，不读取真实标签，也不启动正式训练。`preflight` 会读取真实缓存并执行迷你训练，因此仅在确实需要真实数据预检时使用。

实验不会覆盖 `04_results/final_submission/prediction.npy`。
