# exp_012 统一运行时地基报告（exp_012_framework_base）

## 1. 目标

把 exp_011 已验证的运行时抽象为全框架统一底座，并建立"预注册 → 自动比对 → 判定归档"的决策日志机制，为后续所有候选组件提供统一的三级验证与晋级判定。

## 2. 结构

```
02_experiments/exp_012_framework_base/
  src/y1_fw_lib.py        统一运行时库（复用 exp_011 底座 + 框架级能力）
  src/run_exp012_base.py  骨架驱动（数据契约 + 锚点复现 + 三级验证 + 提交契约 + 决策日志）
  src/run_exp012_retrain.py 训练终点策略决策实验（阶段1）
  src/run_exp012_factory.py 特征工厂准入裁决（阶段2）
  src/run_exp012_zoo.py     模型动物园（阶段3）
04_results/_decision_log/  决策日志（预注册记录，含门槛表、测量值、判定）
```

`y1_fw_lib.py` 提供：标准折叠定义（fold_1/fold_2/shadow）、决策种子集（42/2026/3407）、`PredictionCache`（primary + fallback 多级缓存，指纹与 exp_011 完全一致）、`DecisionLog`（pre_register/verify）、`evaluate_gates`（门槛测量统一入口）、转移衰减表、动物训练器（CatBoost / LGBM 变体）。

## 3. 骨架运行结果（全部预测复用 exp_011 runtime_cache，零新增训练）

| 项 | 本骨架 | exp_011 记录 | 一致性 |
|---|---|---|---|
| 锚点复现（Valid，s42） | 0.092940 | 0.092940 | 精确 |
| 近期专家（Valid，s42） | 0.089657 | 0.089657 | 精确 |
| fold_1 anchor / recent | 0.112031 / 0.110115 | 同左 | 精确 |
| fold_2 anchor / recent | 0.097117 / 0.096467 | 同左 | 精确 |
| 权重选择 | 最佳 Δ=0.001066 → w=0.30 | 同左 | 精确 |
| 影子（3 种子集成） | 0.084882 → 0.085628（+0.000747） | 同左 | 精确 |
| 官方 Valid（3 种子集成） | 增量 -0.000022 | 同左 | 精确 |

决策日志判定：9/12 门槛通过，未通过项为 official_valid_mean_delta、official_valid_late_delta、seed_ensemble_stable——与 exp_011 已知的"3 种子集成在官方 Valid 不稳定"结论一致，机制正确识别。test_anchor_corr 0.939 低于 0.98 门槛属预期：骨架为 Train-only 口径，正式锚点训练至 3161，训练终点不一致所致（阶段 1 已决策并说明）。

## 4. 产物

- `04_results/exp_012_framework_base/`：prediction.npy、fold_results.csv、weight_search.csv、metrics.json、run.log
- `04_results/_decision_log/20260807_195201_skeleton_anchor_recent_blend.json`：完整预注册 + 门槛比对 + 判定

## 5. 结论

统一运行时地基可用：契约校验、锚点复现、三级验证、决策日志全部跑通，且与 exp_011 逐项一致。后续阶段 1~5 全部基于该底座执行。
