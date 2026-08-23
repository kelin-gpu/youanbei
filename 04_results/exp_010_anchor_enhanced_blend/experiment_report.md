# 锚点增强融合实验报告（exp_010）

- 运行模式：`full`
- 状态：`completed_not_promoted`
- 主特征：`legacy_328`（328 维）
- 增量特征：`exp_008 surprise/trend/history_state`（105 维）
- 已选特征集：`enhanced`（启用增强）
- 已选时间策略：`decay_1200`（decay_1200）
- 已选专家轮数：`8`
- 已选融合权重：`0.00`
- TCN 权重：`0.00`（未启用）
- 锚点：`D:\google_dl\book\友安杯\04_results\final_submission\prediction.npy`（只读，未覆盖）
- 官方 Valid 锚点 RankIC：`0.092940`
- 官方 Valid 专家 RankIC：`0.081221`
- 官方 Valid 融合 RankIC：`0.092940`
- 官方 Valid 增量：`+0.000000`（后段 `+0.000000`，最差季度 `+0.000000`）
- 稳定性平台 span：`0.000000`
- Test 与正式锚点平均截面秩相关：`1.000000`
- 是否晋级：`False`
- 预测 SHA-256：`9d322401a2d8fedd38dea66b97578873e721f03eeb93575dbc8bdc2a1aef38e6`

## 1. exp_009 基线复现（Step A）

- 折内 (16, 0.25) 融合与 8 轮锚点相对 exp_009 最大绝对偏差：`0.000000`（容差 5e-4）
- 复现结论：`通过`

## 2. 特征诊断（Step B）

- 比较纯专家（不融合）在 16 轮的折内平均 RankIC：

- `fold_1`：legacy `0.110115` → enhanced `0.110803`，增量 `+0.000688`
- `fold_2`：legacy `0.096467` → enhanced `0.098068`，增量 `+0.001600`
- `fold_3`：legacy `0.085794` → enhanced `0.086177`，增量 `+0.000383`

- 3 折均值增量：`+0.000890`；是否 3 折均正：`True`；门槛：`0.0005`
- 判定：`启用增强 433`

## 3. 时间策略（Step C）

- `fold_1`：full `0.110803` / late `0.132370` → decay `0.113484` / late `0.133316`
- `fold_2`：full `0.098068` / late `0.066304` → decay `0.095043` / late `0.065668`
- `fold_3`：full `0.086177` / late `0.084496` → decay `0.089889` / late `0.085928`

- decay 相对 full 的 3 折均值增量：`+0.001123`；后段增量：`+0.000581`
- 判定：`decay_1200`

## 4. 轮数与权重选择（Step D）

没有候选通过 Train 内稳定性门槛，回退到正式锚点。
- 已选组合：enhanced/decay_1200、8 轮、w=0.00

## 5. 官方 Valid 一次性晋级检查

- 锚点（legacy_328 / full / 8 轮）： mean `0.092940` | late `0.091868` | worst_q `0.059262`
- 近期专家： mean `0.081221` | late `0.076863` | worst_q `0.057405`
- 已选融合： mean `0.092940` | late `0.091868` | worst_q `0.059262`

### 门槛逐项判定

- [ ] `w > 0` → 0.00
- [x] `锚点复现 |valid_anchor - 0.09294016824452567| <= 0.0003` → 0.000000
- [ ] `official_mean_delta >= 0.0003` → +0.000000
- [x] `official_late_delta >= -0.0002` → +0.000000
- [x] `official_worst_quarter_delta >= -0.0015` → +0.000000
- [x] `test_anchor_correlation >= 0.97` → 1.000000
- [x] `稳定性平台 plateau_span < 0.001` → 0.000000

**晋级结论：`False`**

## 6. 与历史实验对比

| 实验 | 官方 Valid RankIC | 线上 RankIC | 说明 |
| --- | --- | --- | --- |
| exp_003（正式提交/锚点） | 0.092940 | 0.108105 | legacy_328 + LambdaRank 8 轮 |
| exp_007 | 0.094446 | 0.109959 | 锚点+近期专家 rank 融合 0.65/0.35 |
| exp_008 | 0.084385 | 0.101942 | 完整因果补全 + 363 特征 + decay |
| exp_009 | 0.093615 | 0.109928 | 锚点+近期专家 0.75/0.25，16 轮 |
| exp_010（本实验） | `0.092940` | 待提交 | enhanced/decay_1200/r8/w0.00 |

## 产物

- `prediction.npy`：本实验完整 Test 结果（晋级时另存 `promoted_candidate.npy`）。
- `expert_prediction.npy`：专家逐时间排名结果。
- `valid_prediction.npy`：官方 Valid 已选融合结果。
- `fold_results.csv` / `blend_fold_results.csv`：折内明细。
- `feature_diagnosis.csv` / `time_strategy.csv`：特征与时间策略诊断。
- `weight_search.csv`：轮数/权重稳定性门槛与最终选择。
- `official_valid_results.csv` / `promotion_gates.csv`：官方 Valid 一次性检查。
- `exp009_reproduction.csv`：exp_009 基线复现核对。

> 本实验不会自动覆盖 `04_results/final_submission/prediction.npy`。
