# 03 模型训练

状态：`archived`（历史流程）。

本目录保存新的统一训练入口，以及已经实际运行过的历史模型实验。

## 当前入口

`03_模型训练.ipynb`只读取`../02_数据处理/processed_data_v1/`，目前已经加入可复现的提分筛选入口：

- Tree允许在固定419列视图中选择`legacy_328`、`numeric_408`或`full_419`，但不重新构造特征；
- Linear固定479列；
- TCN固定40列标准化序列和mask；
- 模型侧不再补值、编码、标准化或构造特征。

模型筛选默认关闭，避免“全部运行”意外开始全量训练。需要训练时设置：

- `Y1_RUN_SCREENING=1`；
- `Y1_SCREEN_PROFILE=quick`：328列与408列；
- `Y1_SCREEN_PROFILE=windows`：328列的1945/1216/730期窗口；
- `Y1_SCREEN_PROFILE=windows_refine`：328列的2188/1702期精细窗口；
- `Y1_SCREEN_PROFILE=blend_refit`配合`Y1_ROUNDS=8`：快速重训全历史与近期专家并搜索Rank融合；
- `Y1_PARAM_PROFILE=tuned_public_best`：使用旧公开榜最佳模型的调优参数做可比复核；
- `Y1_TRAIN_STOCK_CAP=1200`：复现旧主模型每时点等距抽样口径（默认值）；
- `Y1_SCREEN_PROFILE=categories`：419列类别消融。

2026-08-01同口径复核结果保存在`outputs_v2/screening_results.csv`与`screening_blends.csv`：

- 旧最佳参数、每时点1200只股票、全历史8轮精确复现Valid `0.09294015`；
- 1702期专家单独为`0.09113176`；
- 65%全历史与35%近期专家的Rank融合为`0.09444576`，最差季度也小幅改善；
- Test候选保存为`../04_结果/候选提交/y1_blend_full65_recent35.npy`，当前最终提交未覆盖。

缺少`processed_data_v1/READY`时，Notebook会拒绝训练。

## 历史复现

以下Notebook暂时保留用于核对历史结果，不再作为新的正式入口：

1. `y1_linear_baseline/code_y1_baseline.ipynb`：线性基线。
2. `y1_tcn_baseline/code_y1_tcn_window486.ipynb`：第一版 TCN。
3. `y1_rank_pipeline/y1_rank_pipeline.ipynb`：LightGBM LambdaRank 主流水线。
4. `y1_pipeline_v2/03_ensemble.ipynb`：线性、TCN、LightGBM 重训和融合。

`y1_rank_pipeline/y1_rank_pipeline_lib.py` 是 LightGBM Notebook 的内部依赖，不是用户入口。保留它是为了避免把两千多行底层实现塞入一个 Notebook 单元；日常只需要打开 `.ipynb`。

各模型原有 `outputs/` 或 `y1_rank_outputs/` 保存该模型的历史模型权重、Valid/Test 预测和实验报告。正式提交文件统一放在 `../04_结果/最终提交/`。
