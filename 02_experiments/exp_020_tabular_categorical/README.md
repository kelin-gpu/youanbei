# exp_020：cat_5 原生类别纳入 tabular 专家

状态：`historical_component`（已被 exp021 的一致重训方案替代）。

对 exp016 的 tabular 家族做单变量改动：LightGBM 的 `lgbm_rank` / `lgbm_huber` 显式把 9 个类别列（列 408..416，含高基数 cat_5）作为原生类别特征（`categorical_feature`），与 CatBoost 对齐；其余 6 个家族、元头、路由权重全部复用 exp016 既有产物（不重训）。

## 改动

`02_experiments/exp_016_unified_expert_fusion/src/tabular_experts.py` 的 `train_tabular_family`：LightGBM 两个 Dataset 增加 `categorical_feature=list(CATEGORY_INDICES)`。

## 结果（全量 Valid 口径）

| 指标 | 值 |
|---|---|
| 新 tabular 家族独立全量 Valid IC | **0.094474**（旧口径 0.091634 capped） |
| 替换 tabular 后重融合全量 Valid IC | **0.091888**（基准路由 0.091389，Δ +0.0005） |

- tabular 家族独立增益约 +0.003，说明 cat_5 原生类别对 tabular 有效。
- 重融合仅 +0.0005，因为元头/路由权重未重训（仍按旧 tabular 训练），只体现 tabular 家族单点替换的一阶效应。

## 产物

`04_results/exp_020_tabular_categorical/`：`prediction_1.npy`（重新融合的 Test 提交，契约通过）、`metrics.json`、`metadata.json`。不覆盖 `04_results/final_submission/`。

## 运行

```powershell
& 'D:\anaconda\anaconda_data\envs\jingge_ts\python.exe' 02_experiments/exp_020_tabular_categorical/run_exp020.py
```

## 线上反馈

- `prediction_1.npy`（SHA `812F7165...`）线上 RankIC **`0.116252`**（exp016 基线 0.116132，**+0.000120**），为 exp016 之后首个线上正向改动。
- 结论：cat_5 原生类别纳入 tabular 有效且线上可迁移；建议晋级为新的线上最佳，并可继续做「元头 + 路由」一致重训释放完整增益。详见 `04_results/_decision_log/20260818_online_feedback_exp020_cat5_tabular.json`。

## 后续

若 `prediction_1.npy` 线上正向，再做「元头 + 路由」一致重训（复用神经家族，仅重训 tabular + head + router）。
