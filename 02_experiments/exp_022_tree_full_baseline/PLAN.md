# exp_022 纯树基线实验方案（检验「0.12 是否可无泄漏达到」）

## 1. 目标与假设

用户观察到「有人一次提交就到 0.12」，怀疑要么自己的方法有问题、要么对方用了未来数据。
本实验做**决定性检验**：不引入任何新架构，只用一套调好的 GBDT 树模型 + 全量特征 + 9 个类别特征正确编码，纯 walk-forward，看能否逼近 0.12。

- 若纯树能到 ~0.12 → 说明是「方法差距」，深层神经网络堆栈方向走偏，应回头重做树路径。
- 若纯树仍卡在 0.11 出头 → 才需要认真怀疑赛题本身或对手用了非常规手段。

## 2. 关键证据（为什么先怀疑方法而非泄漏）

单因子 RankIC 分析（`01_analysis/outputs/rankic_summary.csv`）显示：最强单因子 num_57 的 IC 仅 **0.058**（train）/ 0.052（valid），正率 87%。**没有任何特征 IC 接近 0.3~0.5**，说明原始特征里不存在「直接写进未来标签」的泄漏特征。而项目长期把 9 个类别特征当数值列、树模型只训 16 轮、纯树基线只用 40 特征——这才是更可能的差距来源。

## 3. 目录与命名

- 实验脚本：`02_experiments/exp_022_tree_full_baseline/run_exp022.py`
- 结果：`04_results/exp_022_tree_full_baseline/`
- 提交文件按用户规则命名 `prediction_1..N.npy`，每变体一份；另有 `models/`、`metrics.json`、`metadata.json`、`fold_results.csv`。
- 不覆盖 `04_results/final_submission/`。

## 4. 环境与数据

- Python：`D:\anaconda\anaconda_data\envs\jingge_ts\python.exe`（torch 2.6 / lgb 4.7 / cb 1.2.10 / xgb 3.2 / scipy 1.15 已装）
- 树特征（419 列）：`03_cache/processed_data_v1/tree/{train,valid,test}_X.npy`
  - 列 0..407 = 408 个数值特征；列 408..416 = 9 个类别（cat_0..cat_8，cat_5=索引 413 高基数 4093）；列 417..418 = 待核验的额外两列（沿用 exp016 现状，不特殊处理）。
- 标签/分组：`03_cache/processed_data_v1/common/{train,valid}_{y,relevance,group_sizes}.npy`；`common/test_{time,stock,group_sizes}.npy`（test 无 y/relevance）
- 切分：train `[486,2918)`、valid `[2918,3161)`、test `[3161,3603)`；提交 `(442,5282)` float32，评估位 2,042,538，非评估位 292,106 填 0.5。

## 5. 变体定义（每个 = 一份提交）

| # | 变体 | 模型 | 类别处理 |
|---|---|---|---|
| 1 | lgbm_native_cat | LightGBM LambdaRank | 9 类别作为 `categorical_feature`（原生类别） |
| 2 | catboost_yetirank | CatBoost YetiRank | 9 类别作为 `cat_features` |
| 3 | xgboost_pairwise | XGBoost rank:pairwise | 9 类别 int 序数 + `enable_categorical` |
| 4 | ensemble_rank | 1+2+3 截面排名取平均 | 三者融合 |
| 5 | lgbm_target_enc | LightGBM LambdaRank | 9 类别做「扩展窗口 target encoding」替换原始类别 |

训练参数（统一）：`learning_rate=0.05`、`num_leaves=63`、早停 `early_stopping=50`、最大轮数 LGBM/XGB 1000、CatBoost 1000、`label_gain=range(64)`、`lambdarank_truncation_level=1024`、`seed=42`。

## 6. 编码与防泄漏规则（本实验的红线）

- **频率编码**：只用 train `[486,2918)` 统计，unseen 填 0。
- **target encoding**：用**扩展窗口**——按时间截面递增，只用「严格早于当前时点」的历史样本算每类别的 y 均值，禁止用同截面或未来标签（防未来泄漏）。
- **早停**：用 valid `[2918,3161)`（含标签）做早停——这是标准 walk-forward，允许。
- **最终 Test 模型**：用 `best_iteration`（×1.1 补偿更多数据）在 train+valid `[486,3161)` 重训后预测 test。
- **绝不加载 test 标签**（数据里也没有）；编码统计与阈值只来自训练侧历史。

## 7. 执行步骤（脚本内两阶段）

- **Phase A（选模）**：train `[486,2918)` 训练 → valid 早停 → 记录 `best_iteration` 与 valid mean RankIC。
- **Phase B（提交）**：train+valid `[486,3161)` 用 best_iteration 重训 → test 分块推理 → 组内 rank → `vector_to_grid` → `validate_prediction` → 存 `prediction_*.npy`。

## 8. 运行命令

```powershell
# 冒烟（合成数据，几秒，验证链路）
& D:\anaconda\anaconda_data\envs\jingge_ts\python.exe 02_experiments/exp_022_tree_full_baseline/run_exp022.py --smoke

# 正式跑（默认 cap=1024，安全；5 变体约 15-30 分钟）
& D:\anaconda\anaconda_data\envs\jingge_ts\python.exe 02_experiments/exp_022_tree_full_baseline/run_exp022.py

# 只跑某几个变体
& D:\anaconda\anaconda_data\envs\jingge_ts\python.exe 02_experiments/exp_022_tree_full_baseline/run_exp022.py --variants 1,4
```

> `--cap 0` 用全量行（质量更好但需 ~25GB 空闲内存）；默认 `--cap 1024` 与 exp016 同口径、内存安全（本机空闲内存偏紧，建议关掉其它占用）。

## 9. 验收标准

1. `fold_results.csv` 含 5 变体 valid mean RankIC 与 best_iteration。
2. 5 个 `prediction_*.npy` 全部通过契约校验（`(442,5282)`、float32、有限、非评估位 0.5、评估位 2,042,538）。
3. 明确回答：纯树基线 valid/线上能否逼近 0.12；相对当前 exp021(0.116568) 的差距。
4. 全程无 test 标签泄漏；`final_submission/` 未被改动。

## 10. 已知坑（接手者必读）

- LightGBM 类别列必须是非负 int（`astype(np.int32)`），负值会被当成缺失。
- CatBoost 用 `cat_features=[408..416]`（0 基索引）；值为 int 即可。
- XGBoost 类别需 `enable_categorical=True` + `feature_types`（'c'/'q'），否则按数值列处理、信号弱。
- 内存：train 全量 6.5M×419 float32 ≈ 10.9GB，务必默认 cap=1024，别直接全量 load。
- 类别列是 float 存的（如 cat_5），务必 `np.rint().astype(np.int32)` 后再用。

## 11. 时间与分工

- 冒烟自检：<1 分钟，已由我完成（编译 + 合成数据冒烟）。
- 正式跑：约 15-30 分钟，属 >10 分钟任务，**由用户本机执行**；我只负责脚本与安全检测。
