# 三模型重训与秩集成实验报告

- 生成时间：2026-08-01 13:28:18
- Train：`[486, 2918)`
- Valid：`[2918, 3161)`
- Test：`[3161, 3603)`
- 线性模型：3 epochs，Train-only 标准化
- TCN：486 期窗口，固定 8 epochs，无 Valid 早停
- LightGBM：固定 1558 轮，无 Valid 早停
- 权重网格：231 组，步长 0.05

## 单模型 Valid RankIC

```csv
model,first_half_rank_ic,second_half_rank_ic,full_rank_ic
linear,0.094542,0.084773,0.089678
tcn,0.105142,0.083234,0.094233
lgbm,0.071455,0.064819,0.068151
```

## 选择结果

- promoted：`False`
- 选定权重（linear / tcn / lgbm）：`0.40 / 0.40 / 0.20`
- 前半窗 RankIC：`0.109618`
- 后半窗 RankIC：`0.092883`
- 完整 Valid RankIC：`0.101285`
- 邻近权重最大波动：`0.001277`
- 前半窗超过最佳单模型：`True`
- 后半窗超过最佳单模型：`True`
- 后半窗高于 0.093940：`False`
- 权重平台稳定：`False`

## 输出

- 未通过门槛，因此本次未生成候选提交文件。
- 权重明细：`ensemble_weight_search.csv`
