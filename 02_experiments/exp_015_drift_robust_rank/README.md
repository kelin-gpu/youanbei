# exp015：DriftRouter-Rank 创新增强版（integrated_v2）

状态：`historical_strong_baseline`（保留为强基线，不是当前提交）。

`build_notebook.py` 是唯一源码，运行后生成 `experiment.ipynb`，内核固定为 `jingge_ts`。本版不新建 exp016，而是在 exp015 内原地融合 `robust_rank_348`、CatBoost/YetiRank、多尺度 patch、16 个原型、因果掩码预训练、OOF 正交残差、漂移 top-2 路由及专家分歧回缩。

新增缓存和结果分别隔离到：

- `03_cache/exp_015_drift_robust_rank/integrated_v2/`
- `04_results/exp_015_drift_robust_rank/integrated_v2/`

## 训练安全闸门

Notebook 默认模式是 `smoke`。任何会更新真实模型参数的入口都调用统一训练授权检查；必须同时满足：

```powershell
$env:DSCR_EXP015_MODE = "full"
$env:DSCR_EXP015_ALLOW_TRAINING = "YES"
```

否则 LightGBM、CatBoost、PyTorch 残差专家、路由器以及 full 主流程都会立即终止。本轮禁止设置上述组合。

`DSCR_EXP015_STAGE` 支持 `all|features|pretrain|experts|router|final`，默认 `all`，为以后显式授权训练时断点执行预留。

## 本轮允许的检查

### Smoke

```powershell
$env:DSCR_EXP015_MODE = "smoke"
Remove-Item Env:DSCR_EXP015_ALLOW_TRAINING -ErrorAction SilentlyContinue
```

仅使用合成数据，检查截面秩 ties、5/20/60 因果窗口、随机初始化前向、loss/backward 计算图、参数未更新、原型排列等变性、正交化、top-2 路由、专家参与率、不确定性回缩、缓存指纹、原子写入和训练保护。

### Preflight dry-run

```powershell
$env:DSCR_EXP015_MODE = "preflight"
Remove-Item Env:DSCR_EXP015_ALLOW_TRAINING -ErrorAction SilentlyContinue
```

读取 6 个真实时间点并构建 20 维秩特征；仅构造 LightGBM Dataset 和 CatBoost Pool，不调用训练。随机初始化神经模块完成真实 sequence/patch/mask/原型前向、损失、路由和融合检查。`preflight/contract_sample.npy` 只验证 Test 输出契约，不是候选文件。

## 本轮交付状态

通过后应存在：

- `integrated_v2/smoke/smoke_report.json`
- `integrated_v2/preflight/preflight_report.json`
- `integrated_v2/static/static_check_report.json`

preflight 报告状态必须为 `IMPLEMENTED_AND_DRY_RUN_PASSED`，且 `training_performed=false`。本轮不产生 exp015 根目录 `prediction.npy`，不运行 full，不生成可提交预测，也不覆盖 `04_results/final_submission/prediction.npy`。
