# Results

每个实验的结果写入与实验编号同名的子目录。完整结果统一使用 `prediction.npy`、`metrics.json`、`metadata.json`、`model.*` 和 `experiment_report.md`。

所有实验方法、本地成绩、线上提交成绩和后续判断统一维护在项目根目录 `README.md` 的“成绩与实验记录”一节。

`historical_metrics_only` 表示旧流程只保留了验证指标；运行对应的新 Notebook 后会生成标准预测和模型文件。

`final_submission/prediction.npy` 是当前正式提交。任何实验都不得自动覆盖它。

当前状态：

- 线上最佳候选为 `exp_024b_retrieval_exploratory/prediction_1.npy`，用户记录线上 RankIC `0.120847`；
- 栈基准为 `exp_021_retrain_head_router/prediction_1.npy`，线上 RankIC `0.116568`；
- `final_submission/prediction.npy` 已经用户授权晋级为 exp024b，线上 RankIC `0.120847`，SHA-256 `6ff796c7...`；
- 机器可读实验登记见 `04_results/experiment_registry.csv`，项目状态见根目录 `project_status.json`。
