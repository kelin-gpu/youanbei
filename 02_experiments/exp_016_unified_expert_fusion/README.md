# exp_016_unified_expert_fusion

状态：`historical_formal_source`（正式提交来源；当前栈基准为 exp021，线上最佳候选为 exp023h）。

exp016 是一个可审计的统一多专家排序系统。它把表格排序、双轴时序—截面建模、因果时频特征、稀疏关系图、本地无标签表征学习、多目标排序头和市场状态路由组合起来。Python 模块是唯一事实源，`experiment.ipynb` 只负责选择模式并调用 CLI。

## 快速结论

- 当前状态：`full` 已完成，`metadata.json` 的状态为 `full_completed`。
- 历史线上记录：RankIC `0.116132`（用户记录日期 2026-08-16），较 exp015 的 `0.110779` 提升 `+0.005353`；当前项目最佳候选为 exp023h（`0.119660`）。
- 官方 Valid：mean RankIC `0.0904873358`，243 个截面、248,832 行；该验证口径与 exp015 不同，不直接横向比较。
- 候选提交：`04_results/exp_016_unified_expert_fusion/full/prediction.npy`。
- 受保护的 `04_results/final_submission/prediction.npy` 未被覆盖。

## 运行方式

先在项目根目录 `D:\google_dl\book\youanbei` 打开 PowerShell，并激活包含 PyTorch、LightGBM、CatBoost、XGBoost 和 SciPy 的实验环境。以下命令中的 `python` 可以替换成环境的完整 Python 路径。

### 验收模式

下面三种模式都不会执行优化器更新、树模型训练或生成提交预测：

```powershell
$env:DSCR_EXP016_MODE = "static"
python -m 02_experiments.exp_016_unified_expert_fusion.run_exp016

$env:DSCR_EXP016_MODE = "smoke"
python -m 02_experiments.exp_016_unified_expert_fusion.run_exp016

$env:DSCR_EXP016_MODE = "preflight"
python -m 02_experiments.exp_016_unified_expert_fusion.run_exp016
```

验收结果分别写入：

```text
04_results/exp_016_unified_expert_fusion/static/static_check_report.json
04_results/exp_016_unified_expert_fusion/smoke/smoke_report.json
04_results/exp_016_unified_expert_fusion/preflight/preflight_report.json
```

其中：

- `static` 编译所有 Python 模块和 Notebook 代码，并检查训练保护。
- `smoke` 用随机数据验证各专家的前向/反向接口、排名、路由和输出契约，但不调用 `optimizer.step()`。
- `preflight` 使用真实数据检查缓存、因果历史、时频分解、图结构、树模型容器、相关性标签和资源预算，但不训练。

### 真实训练

真实训练必须显式设置两个环境变量：

```powershell
$env:DSCR_EXP016_MODE = "full"
$env:DSCR_EXP016_ALLOW_TRAINING = "YES"
python -m 02_experiments.exp_016_unified_expert_fusion.run_exp016
```

如果只设置其中一个，训练会被 `require_training()` 拒绝。`full` 只有在自监督预训练、三个严格 OOF 折、官方验证、七个家族预测、动态路由和最终输出契约全部通过后，才会写入自身目录的 `prediction.npy`。

### Notebook

`experiment.ipynb` 默认使用 `RUN_FULL = False`，即运行 `preflight`。只有人工将它改为 `True`，Notebook 才会设置 `DSCR_EXP016_ALLOW_TRAINING=YES` 并执行 `full`。代码更新后建议重启 `jingge_ts` 内核，避免复用旧模块。

## 整体架构

```text
DataContext
  ├─ 419维表格特征 + 40维序列特征
  ├─ 因果240步历史窗口
  └─ Train / Valid / Test 时间边界
          │
          ├─ exp015 Anchor：LightGBM LambdaRank 基线
          ├─ Tabular：LightGBM LambdaRank + LightGBM Huber
          │           + CatBoost YetiRank + XGBoost pairwise
          ├─ Dual Axis：多尺度时序卷积 + 截面原型交互
          ├─ Time-Frequency：因果趋势/周期分解 + 时频卷积
          ├─ Relational Graph：稀疏KNN + 类别上下文 + Lead-Lag消息传递
          └─ Foundation Representation：无标签预训练编码器 + 监督预测头
          │
          ├─ OOF 六家族预测
          ├─ MultiObjectiveRankHead：相关性、成对排序、尾部和置信度目标
          └─ StateRouter：按时间截面动态分配七家族权重
          │
          └─ 截面排名融合 → 输出矩阵 → 契约校验
```

最终融合不是直接平均原始分数，而是先对每个专家在每个截面内做排名，再按路由器给出的截面权重加权，最后再次做截面排名。每个家族都有正的最小权重，避免系统完全依赖单一模型。

## 数据边界与防泄漏

- 自监督预训练：`[0,2918)`，只使用序列，不使用标签。
- OOF Fold 1：`[486,1459)` 训练，`[1459,1945)` 预测。
- OOF Fold 2：`[486,1945)` 训练，`[1945,2432)` 预测。
- OOF Fold 3：`[486,2432)` 训练，`[2432,2918)` 预测。
- 官方 Valid：`[486,2918)` 训练，`[2918,3161)` 预测。
- 最终监督训练：`[486,3161)`；Test：`[3161,3603)`。

OOF 的每一折都只使用预测区间之前的标签训练。`DataContext` 永不加载 Test 标签。训练阶段默认 `stock_cap=1024` 用于控制资源；最终 Test 推理使用全部 5,282 只股票，不会因为训练采样而把官方评估位置错误填成 `0.5`。

## 代码地图

| 文件 | 作用 |
|---|---|
| `run_exp016.py` | CLI 入口，读取环境变量并调用实验管线 |
| `config.py` | 时间边界、家族、权重、训练授权和路径配置 |
| `src/pipeline.py` | `static`、`smoke`、`preflight`、`full` 四种模式 |
| `src/full_pipeline.py` | 自监督、OOF、官方验证、最终训练和提交生成的主流程 |
| `src/data_context.py` | 只读数据访问、分组切片、因果历史窗口 |
| `src/tabular_experts.py` | 四个异构树模型及其组内排名融合 |
| `src/dual_axis.py` | 双轴时序—截面专家 |
| `src/time_frequency.py` | 训练期周期库、因果分解和时频专家 |
| `src/relational_graph.py` | 稀疏股票关系图和图专家 |
| `src/self_supervised.py` | 无标签编码器、预训练任务和表征头 |
| `src/multi_objective_head.py` | 六家族预测的多目标排序头 |
| `src/state_router.py` | 截面状态特征和动态家族路由 |
| `src/training.py` | 仅允许 full 模式调用的训练循环与 checkpoint |
| `src/ranking.py` | 组内排名、RankIC 和静态/动态融合 |
| `src/prediction_contract.py` | `(442,5282)` 输出矩阵和非评估位置校验 |
| `src/artifacts.py` | 原子写入、SHA-256 和跨平台模型文件保存 |

## 主要产物

```text
03_cache/exp_016_unified_expert_fusion/
  ├─ checkpoints/                  # 自监督、OOF、官方Valid和final模型
  ├─ oof_predictions/              # OOF/官方Valid预测、家族矩阵和动态权重
  └─ spectral/final_period_library.npy

04_results/exp_016_unified_expert_fusion/full/
  ├─ family_*.npy                  # 七个家族的 Test 预测
  ├─ dynamic_weights.npy           # 每个截面的七家族权重
  ├─ prediction.npy                # 候选提交矩阵
  ├─ submitted_prediction.npy      # 写入前的提交矩阵副本
  ├─ submission_*.npy              # 按 SHA-256 命名的不可变副本
  ├─ metadata.json                 # 当前完成状态、验证结果和输出哈希
  ├─ full_report.json              # full 阶段摘要
  ├─ run_manifest.json             # 数据 manifest 和运行信息
  └─ online_feedback_template.json # 线上成绩回填模板
```

`metadata.json` 和 `full_report.json` 是当前 full 成功状态的主要依据；若目录中存在 `failure.json`，它表示某次历史失败尝试，不能替代当前的成功元数据。

## 当前结果与输出契约

- 官方 Valid：243 个截面、248,832 行，mean RankIC `0.09048733578256966`。
- 线上 RankIC：`0.116132`，线上提交日期为 2026-08-16。
- 提交文件 SHA-256：`5721e5fa325ecce624755da65db6e0245f2e79857f14f6d61e4fed9d9c83c524`。
- 剪枝候选 `prediction_pruned_no_dual_axis.npy`（SHA `CBD3B3FA...`，删除 dual_axis）已提交线上，RankIC `0.115654`（相对原版 −0.000478）。本地 capped Valid 曾估 +0.0028，线上为负，属本地→线上错配；**dual_axis 剪枝线上负向，T2.1 精简重训暂缓**。详见 `04_results/_decision_log/20260817_online_feedback_exp016_pruned.json`。
- 输出 shape：`(442,5282)`。
- dtype：`float32`。
- 全部值有限。
- 评估位置：`2,042,538`。
- 非评估位置：`292,106`，全部为 `0.5`。
- `formal_submission_overwritten: false`。

正式提交目录仍然受保护。exp016 的 `prediction.npy` 是历史候选；当前线上最佳候选为 exp023h，但不会自动替换 `04_results/final_submission/prediction.npy`。
