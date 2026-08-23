# exp_021：元头 + 路由一致重训（categorical tabular）

状态：`active_stack_baseline`（当前栈基准，线上 RankIC `0.116568`）。

exp_020 只替换了 tabular 家族（LightGBM 9 类别列作为原生类别），head/router 仍按旧 tabular 训练，重融合仅 +0.0005（tabular 独立 +0.003）。本实验在 OOF 六家族矩阵上用 categorical tabular 重算 states 后，**端到端重训 multi_objective_head 与 state_router**，释放完整增益。

## 复用（不重训）

- 神经家族 checkpoint（OOF 折 / official_valid / final）
- `family_matrix.npy` 中除 tabular 外的 5 个神经/锚点 OOF 列
- exp016 full 的 Test 家族预测（除 tabular）；`official_valid_first_six.npy`（capped valid 前六家族）

## 重训

- 5 套 tabular：3 个 OOF 折 + official_valid + final（LightGBM categorical）
- `multi_objective_head`、`state_router`（与 exp016 相同的 loss / epochs）

## 阶段

- `fit`：重训 tabular(OOF×3 + official_valid) + head + router，输出 capped valid mean IC（与 exp016 0.090487 同口径对照）
- `submit`：重训 final tabular，生成 Test 提交 `prediction_1.npy`（契约校验）
- `all`：两者都做

## 运行（预计 40–50 分钟，>10 分钟由你本机执行）

```powershell
# 建议先跑 fit 看 capped valid 是否提升
$env:DSCR_EXP016_MODE = "full"
$env:DSCR_EXP016_ALLOW_TRAINING = "YES"
& 'D:\anaconda\anaconda_data\envs\jingge_ts\python.exe' 02_experiments/exp_021_retrain_head_router/run_exp021.py fit

# fit 达标后再生成提交
& 'D:\anaconda\anaconda_data\envs\jingge_ts\python.exe' 02_experiments/exp_021_retrain_head_router/run_exp021.py submit
```

也可一次跑 `run_exp021.py all`。

## 安全检测（已完成）

- `py_compile` 通过（PY_COMPILE_OK）
- 训练保护有效：非 full 模式被 `require_training` 拒绝（GUARD_OK）
- head/router 训练链路小数据冒烟通过（SMOKE_HEAD_ROUTER_OK）

## 产物

- `04_results/exp_021_retrain_head_router/`：`metrics_fit.json`（fit 阶段）、`metrics.json` + `prediction_1.npy`（submit 阶段）、`metadata.json`
- `03_cache/exp_021_retrain_head_router/`：重训后的 `multi_objective_head.pt`、`state_router.pt`
- 不覆盖 exp016 任何产物与 `04_results/final_submission/`

## 线上反馈

- `prediction_1.npy`（SHA `ACD8AE24...`）线上 RankIC **`0.116568`**（exp020 0.116252、exp016 0.116132），为当前线上最佳。
- fit 阶段 capped valid +0.000485（0.090972 vs 0.090487）；线上相对 exp020 +0.000316、相对 exp016 +0.000436。
- 按用户规则（未大于 0.12 不晋级正式提交），`final_submission` 保持 exp016 不变；距目标 0.12 差 0.003432。详见 `04_results/_decision_log/20260818_online_feedback_exp021_head_router.json`。
