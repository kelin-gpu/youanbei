# exp_011 数据与项目审计报告

## 1. 原始数据 data.z

- 存在：`True`，大小：`3250274528` 字节（3.03 GiB）
- SHA-256：`a426a7078097e8d970c2f27a30a49b3122a8a0ea7c4c05f35938d5f568cfd04c`
- 与 manifest 期望一致：`True`

## 2. 缓存 processed_data_v1

- READY 存在：`True`；manifest 存在：`True`
- manifest SHA-256：`f7c4076de6e3ae7d631554df5a15f69f50d7e8f676249fb6d2d4cf71ccec8c6f`；与 READY 记录一致：`True`
- 状态：`ready`；legacy 兼容校验：`{"status": "passed", "feature_count": 328, "valid_max_abs_error": 5.957341553397555e-08, "test_max_abs_error": 5.960239524149813e-08}`

## 3. 官方时间边界

- train_start_idx=486：`486`；valid_start_idx=2918：`2918`；test_start_idx=3161：`3161`
- T=3603：`3603`；S=5282：`5282`
- 全部匹配官方划分：`True`

## 4. 监督样本核查

（基于 data.z 原始 mask_x / mask_y / y1 精确统计）


### Train [486, 2918)

- mask_x 有效数：`7,354,184`
- mask_y 有效数：`6,489,099`
- y1 有限数：`6,489,099`
- 监督样本数（mask_x & mask_y & finite(y1)）：`6,489,099`
- 每时间点有效股票数：min=`1208`，max=`3815`，median=`2459.5`，mean=`2668.2`

### Valid [2918, 3161)

- mask_x 有效数：`1,089,602`
- mask_y 有效数：`982,972`
- y1 有限数：`982,972`
- 监督样本数（mask_x & mask_y & finite(y1)）：`982,972`
- 每时间点有效股票数：min=`3813`，max=`4307`，median=`4033.0`，mean=`4045.2`

### Test [3161, 3603)

- mask_x 有效数：`2,219,158`
- mask_y 有效数：`2,042,538`
- y1 有限数：`0`
- 监督样本数（mask_x & mask_y & finite(y1)）：`0`
- 每时间点有效股票数：min=`4305`，max=`4902`，median=`4651.0`，mean=`4621.1`

缓存行数与原始 mask 统计交叉验证一致：`True`（train 6,489,099 / valid 982,972 / test 2,042,538）

### 每时间点有效股票数量汇总

| 区间 | min | max | median | mean |
|---|---|---|---|---|

| Train | 1208 | 3815 | 2459.5 | 2668.2 |

| Valid | 3813 | 4307 | 4033.0 | 4045.2 |

| Test | 4305 | 4902 | 4651.0 | 4621.1 |

## 5. legacy_328 特征复现核查

- 特征数：`328`（manifest `legacy_numeric_prefix=328`）
- 特征顺序：按 `40 原始数值 + 20 截面排名 + 80 滞后 + 60 滚动均值 + 60 滚动标准差 + 60 滚动变化量 + 4 滞后可用性 + 3 历史覆盖率 + 1 存续时间` 排列，结构校验：`{'raw_40': True, 'rank_20': True, 'lag_80': True, 'roll_mean_60': True, 'roll_std_60': True, 'roll_change_60': True, 'lag_available_4': True, 'coverage_age_4': True, 'total_328': True}`
- 训练 shape：`[6489099, 419]`；验证 shape：`[982972, 419]`；测试 shape：`[2042538, 419]`
- 前 4096 行有限值检查：`{'train': True, 'valid': True, 'test': True}`
- 与历史 328 维视图最大绝对误差：valid `5.957e-08`，test `5.960e-08`（<1e-7，判定通过）
- 因果性：滞后/滚动/覆盖率/存续时间全部只使用当前时点及以前信息（与 exp_003 一致）；缓存构建时已校验，本次直接复用缓存。

## 6. 既有测试预测模型的训练终点核查

> 结论：当前四个已提交实验生成 Test 预测时，训练数据均延伸至 3161（包含官方 Valid 标签），并非只训练到 2918。

| 实验 | 测试模型实际训练区间 | 是否包含 Valid 标签 | 证据 |
|---|---|---|---|

| exp_003 | [486 或近期窗口起点, 3161) | 是 | run_pipeline 中 final_train 使用 (TRAIN_START, TEST_START) 区间，即 [486, 3161)，包含 Valid 标签；随后用 official_best_iteration 在测试集预测 |

| exp_006 | [486 或近期窗口起点, 3161) | 是 | 定稿单元 final_start = max(TRAIN_START, VALID_STOP - window)，valid_begin = VALID_START，训练数据拼接 train + valid 至 3161 |

| exp_007 | [486 或近期窗口起点, 3161) | 是 | 最终近期专家 final_stop = VALID_STOP = 3161，训练 train[1459,2918) + valid[2918,3161)；锚点直接读取 final_submission（exp_003 产物，训练至 3161） |

| exp_009 | [486 或近期窗口起点, 3161) | 是 | 模型元数据 model_recent.metadata.json：train_start=1459, train_stop=3161, rounds=16；锚点读取 final_submission |


exp_009 模型元数据：`train_start=1459, train_stop=3161, rounds=16`；exp_007 近期专家同样训练至 3161；exp_003 定稿模型按 `TRAIN_START..TEST_START` 重训至 3161。

## 7. 审计结论

- 数据完整性：data.z 与 processed_data_v1 全部 SHA-256 校验通过；legacy_328 特征视图与历史缓存一致（最大绝对误差 <1e-7）。
- 官方时间边界、监督样本口径均与 README 一致。
- 现有测试预测模型（exp_003/006/007/009）的训练终点均为 3161（含 Valid 标签）。
- 本次实验将严格区分 Train-only（至 2918）与 Train+Valid 重训（至 3161）两条测试预测路径，并通过重训模拟评估两种路径的优劣。
