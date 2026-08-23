# 锚点与近期专家走步融合实验报告

- 运行模式：`full`
- 状态：`submitted_online_best`
- 特征视图：`legacy_328`
- 锚点：`D:\google_dl\book\友安杯\04_results\final_submission\prediction.npy`
- 近期窗口：`1702` 期
- 选中近期轮数：`16`
- 选中近期权重：`0.25`
- 官方 Valid 锚点 RankIC：`0.092940`
- 官方 Valid 融合 RankIC：`0.093615`
- 官方 Valid 增量：`+0.000675`
- Test 与正式锚点平均截面秩相关：`0.993295`
- 线上 RankIC：`0.109928`
- 是否晋级：`True`
- 预测 SHA-256：`8562706cc3a210db7ad2a067852050e22dcc32c8564f9347011d6fccafeef48c`

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

## 线上结果

用户于 2026-08-04 提供线上 RankIC `0.109928`。该成绩超过 `exp_003_lgbm_rank` 的 `0.108105`，是当前线上新高；正式提交目录尚未替换。
