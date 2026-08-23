# 锚点与近期专家走步融合实验报告

- 运行模式：`preflight`
- 状态：`preflight_only`
- 特征视图：`legacy_328`
- 锚点：`D:\google_dl\book\友安杯\04_results\final_submission\prediction.npy`
- 近期窗口：`1702` 期
- 选中近期轮数：`16`
- 选中近期权重：`0.15`
- 官方 Valid 锚点 RankIC：`0.092940`
- 官方 Valid 融合 RankIC：`0.093540`
- 官方 Valid 增量：`+0.000600`
- Test 与正式锚点平均截面秩相关：`1.000000`
- 是否晋级：`False`
- 预测 SHA-256：`9d322401a2d8fedd38dea66b97578873e721f03eeb93575dbc8bdc2a1aef38e6`

## 选择说明

选择达到最佳平均增益 95% 范围内的最低近期权重。

## 产物

- `prediction.npy`：本实验完整 Test 结果。
- `expert_prediction.npy`：近期专家逐时间排名结果。
- `valid_prediction.npy`：官方 Valid 选中融合结果。
- `fold_results.csv`：各折、轮数与权重的明细。
- `weight_search.csv`：稳定性门槛和最终选择。
- `official_valid_results.csv`：官方 Valid 一次性检查。

> 本实验不会自动覆盖 `04_results/final_submission/prediction.npy`。
