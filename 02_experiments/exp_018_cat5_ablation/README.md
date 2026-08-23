# exp_018：cat_5 四种处理消融

状态：`historical_rejected`（消融完成，轻量模型未晋级）。

T1.2 任务：对高基数类别特征 cat_5（树视图第 413 列，train 内 4093 个取值）做四种处理消融，去风险并判断是否值得纳入主线。

## 四种处理

| 处理 | 说明 | 提交文件 |
|---|---|---|
| native | cat_5 作为 LightGBM categorical 特征（原始整数） | `prediction_1.npy` |
| frequency | cat_5 → 训练侧频率编码（unseen 填 0） | `prediction_2.npy` |
| unknown_bucket | 未见取值显式映射到单一 UNK 桶 | `prediction_3.npy` |
| remove | 完全移除 cat_5（baseline） | `prediction_4.npy` |

## 口径

- 数值主线：legacy_328 = tree 视图前 328 列。
- 训练：LightGBM LambdaRank（64 级 relevance，`lambdarank_truncation_level=1024`），16 轮，`learning_rate=0.05, num_leaves=31, seed=42`，Train-only。
- 训练用 `stock_cap=1024` 控制资源（与 exp016 一致）；cat_5 词表/频率在全量 train 上统计。
- 评估：统一全量 valid（982,972 行）mean RankIC；晋级阈值 ΔIC ≥ 0.0005（相对 remove）。

## 运行

```powershell
& 'D:\anaconda\anaconda_data\envs\jingge_ts\python.exe' 02_experiments/exp_018_cat5_ablation/run_exp018.py
```

## 产物

`04_results/exp_018_cat5_ablation/`：`prediction_1..4.npy`、`model_*.txt`、`fold_results.csv`、`metrics.json`、`metadata.json`。不覆盖 `04_results/final_submission/`。

## 线上反馈

- `prediction_1.npy`（native cat_5）线上 RankIC `0.107010`（SHA `A63D901F...`）。
- 结论：native cat_5 本地全量 Valid +0.0069、线上 0.107010 接近 exp003(0.108105)，证明 cat_5 原生类别处理有效，建议纳入 tabular 专家；但本轻量 16 轮模型不晋级为线上最佳（当前最佳 exp016 = 0.116132）。详见 `04_results/_decision_log/20260817_online_feedback_exp018_cat5_native.json`。
