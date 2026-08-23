# exp_019：Conservative Router 保守路由

状态：`historical_rejected`（保守路由线上负向，不再作为主线）。

T1.4 任务：用 `w = (1-γ)·w_static + γ·w_router` 收缩自由路由，为 `exp015_anchor + tabular` 保底合计 ≥60%、深度家族单家 ≤10%，降低路由过拟合。本轮为离线后处理（零训练）。

## w_static（合计 1.0）

| 家族 | 权重 |
|---|---|
| exp015_anchor | 0.40 |
| tabular | 0.20 |
| dual_axis | 0.08 |
| time_frequency | 0.06 |
| relational_graph | 0.06 |
| foundation_representation | 0.08 |
| multi_objective_rank | 0.12 |

anchor+tabular 合计 0.60；深度家族（dual_axis/time_frequency/relational_graph/foundation_representation）单家 ≤0.10。

## 方法

1. 对 T0.1b 产出的全量 Valid 动态权重逐 γ ∈ {0, 0.25, 0.5, 0.75, 1.0} 做收缩 + anchor/tabular ≥60% 硬地板 + 归一化。
2. 用全量 Valid 七家族预测重融合，比较 mean RankIC。
3. 选择最优 γ，用 exp016 full 的 Test 七家族预测 + Test 动态权重重融合生成提交。

## 运行

```powershell
& 'D:\anaconda\anaconda_data\envs\jingge_ts\python.exe' 02_experiments/exp_019_conservative_router/run_exp019.py
```

## 产物

`04_results/exp_019_conservative_router/`：`conservative_weights.csv`、`prediction_1..n.npy`、`metrics.json`、`metadata.json`。不覆盖 `04_results/final_submission/`。

## 线上反馈

- `prediction_1.npy`（γ=0 纯静态，anchor+tabular=60%）线上 RankIC `0.114914`（SHA `064780EE...`）。
- 结论：保守路由本地全量 Valid +0.0013、线上 −0.0012（0.114914 < exp016 0.116132），属本地→线上错配，路由保底约束不晋级。详见 `04_results/_decision_log/20260817_online_feedback_exp019_conservative_router.json`。
