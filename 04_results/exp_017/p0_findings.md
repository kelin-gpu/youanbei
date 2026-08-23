# exp016 P0 归因与剪枝审计报告（2026-08-17）

## 一、口径审计结论（T0.1）

- exp016 官方 Valid 只有 `248,832` 行 = `243 组 × 1024`，是 `stock_cap=1024` 下每个截面**均匀抽样**（`np.linspace`）得到的子集；而 exp015 官方 Valid 为全量 `982,972` 行。
- 因此 exp016 的 `0.090487` 与 exp015 的 `0.094341` **不可直接比较**。
- 已重建该 1024 抽样子集并重算 final blend 的 mean IC = `0.09048734`，与 metadata 完全一致，证明口径重建与 RankIC 实现均正确。
- **要得到可比较的全量 Valid 数值**，需用现有 `checkpoints/official_valid/` + `multi_objective_head.pt` + `state_router.pt` 做一次全量 inference 重跑（无需训练）。这记为后续任务 T0.1b。

## 二、家族贡献归因（T0.2）

一致信号（两个验证集都指向）：

| 结论 | 证据 |
|---|---|
| `dual_axis` 是最弱/负贡献家族 | 独立 IC 最低（capped valid 0.0727）；leave-one-out 删除后 capped valid +0.0028、OOF +0.0005 |
| `foundation_representation` 正贡献 | 删除后 capped valid -0.0016、OOF -0.0015 |
| `tabular` 正贡献 | 删除后 capped valid -0.0012、OOF -0.0007 |
| `multi_objective_rank`（元头）正贡献 | OOF 删除后 -0.0022（贡献最大） |

有分歧信号：`exp015_anchor`（OOF 负、capped valid 弱正）、`time_frequency`（OOF 负、capped valid 弱正）——不建议本轮剪枝。

## 三、稳定性指标（T0.3）

capped 官方 Valid 逐时间 RankIC：mean `0.0905`、std `0.1015`（波动大于均值）、正 IC 比例 `83.1%`、最差截面 `-0.166`、最好 `0.373`。整体稳定但时间波动较大。

## 四、资源审计（T0.4）

- 神经模型总参数仅 `207,973`；`dual_axis` 62,533 是最大神经家族（却最弱）。
- 树模型文件：lgbm_rank 81.9KB、lgbm_huber 75.8KB、catboost 33.9KB、xgboost 80.2KB、anchor 141KB。
- checkpoint 总大小 `9.75 MB`。

## 五、剪枝候选（T1.3）

- 产物：`prediction_pruned_no_dual_axis.npy`（删除 dual_axis、剩余 6 家族权重归一化）。
- 契约通过：shape `(442,5282)`、float32、全部有限、非评估位 292,106 个全为 0.5。
- 与原始提交的 Test 截面秩相关 `0.999228`（改动微小，符合单变量剪枝预期）。
- 本地 capped valid 估计增益 `+0.0028`（6 家族口径），**未线上验证**，按项目规则需人工晋级与线上校准。

## 六、后续任务

1. **T0.1b**：全量 Valid inference 重跑，得到与 exp015 同口径的 exp016 数值。
2. **T1.2**：cat_5 四种处理消融（valid 有 54,307、test 有 353,411 个 unseen，风险显著）。
3. **T1.4**：Conservative Router（anchor+robust-tree 最低权重 ≥60%）。
4. 视剪枝候选线上反馈决定是否做 **T2.1 精简专家池重训**（去掉 dual_axis 后端到端重训 head/router）。
