# 03 集成评估方案：线性 + TCN + LGBM 秩加权集成

> 新会话使用说明：本文件是完整任务简报。项目根目录 `D:\google_dl\book\友安杯`，
> 工作目录 `03_模型训练/y1_pipeline_v2/`，conda 环境 `jingge_ts`。
> 先读项目根目录 `README.md` 与本文件，再动手写 `03_ensemble.ipynb`。

## 目标

把三份现成模型（线性 0.0897 / TCN 0.0916 / LGBM 0.0929）做**逐时点截面秩加权集成**。
权重在官方 Valid 窗口 [2918, 3161) 的 243 个时点上拟合，应用到测试集三份现成预测，
产出 `outputs/y1_ensemble.npy`。预期 +0.002~0.005。

## 已有资产（全部就位，不要重建）

| 资产 | 路径 | 说明 |
|------|------|------|
| 解压缓存 | `03_模型训练/y1_pipeline_v2/.cache_dp_v2/payload.pkl` | 9.1 GB mmap 缓存，已建好 |
| 测试预测×3 | `03_模型训练/y1_linear_baseline/outputs/baseline_y1.npy`、`03_模型训练/y1_tcn_baseline/outputs/tcn_y1.npy`、`03_模型训练/y1_rank_pipeline/y1_rank_outputs/y1_best.npy` | 均 (442, 5282) float32 |
| 线性模型 | `03_模型训练/y1_linear_baseline/outputs/linear_sgd_model_y1.npz` | 权重+bias |
| TCN 模型 | `03_模型训练/y1_tcn_baseline/outputs/tcn_y1_model.pt` | PyTorch |
| LGBM 模型/清单 | `03_模型训练/y1_rank_pipeline/y1_rank_outputs/{model.txt, feature_manifest.json}` | |
| 评估工具函数 | `03_模型训练/y1_pipeline_v2/y1_pipeline_v2.ipynb` 第 4~5 节 | `eval_mask` / `cross_sectional_rank` / `per_time_rank_ic` / `ensemble_scores`，可直接抄 |
| 数据访问层 | `03_模型训练/y1_rank_pipeline/y1_rank_pipeline_lib.py` | `CompetitionData`，`PipelineConfig(data_path="../../data.z", cache_dir=".cache_dp_v2")` |

## 关键事实（已验证，直接用）

- 测试集 [3161, 3603) 的 Y1 **全为 NaN**，本地无法给测试预测打分；权重只能在 Valid 拟合
- 掩码口径：评估位 = `mask_x & mask_y`（测试）/ 再 `& finite(y1)`（Valid）；`mask_y ⟹ mask_x`
- 三模型测试预测的出分集合是嵌套的：TCN ⊇ 线性 ⊇ 评估集 = LGBM 出分集；
  对齐规则 = 只在评估掩码上取分、各自截面百分位秩化、再加权
- 官方验收锚点：评估位 2,042,538、非评估位中性 0.5 共 292,106（集成输出必须复现这两个数）
- LGBM 官方 Valid IC 0.0929 的模型在 [486,2918) 训练、Valid 早停，best_iteration 见 manifest
  `official_validation.best_iteration`（报告值 8）

## 执行步骤

### Step 1：环境（分钟级）
- 从 `03_模型训练/y1_pipeline_v2/` 启动，内核 `jingge_ts`
- `CompetitionData` 复用 `.cache_dp_v2` 缓存，秒级就绪

### Step 2：重建三模型的 Valid 窗口预测 (243, 5282)

| 模型 | 做法 | 成本 |
|------|------|------|
| 线性 | 读 `code_y1_baseline.ipynb` 确认标准化口径（feature_mean/std 的拟合区间与特征范围），用 npz 权重对 Valid 逐时点出分 | 分钟级 |
| TCN | 加载 .pt，按 `code_y1_tcn_window486.ipynb` 的 `predict_stock_indices` 逻辑推理；Valid 起点前需 486 期 lookback（正好用到 Train 末尾） | ~10 分钟 GPU |
| LGBM | 用 lib 的 `build_feature_matrix`（train [486,2918) cap=1200 + valid [2918,3161)）+ `train_ranker`（winning_lightgbm_params）出 Valid 预测。**不能用 model.txt 直接预测**（它见过 Valid 数据） | ~30~60 分钟 |

### Step 3：单模型复现校验（ sanity gate，不过则停）
分别算三份 Valid 预测的逐时点 RankIC 均值，须逼近 0.089678 / 0.091556 / 0.092940。
对不上先查预处理口径，不许往下走。

### Step 4：权重搜索（243 点，防过拟合）
- 逐时点三模型截面百分位秩 → 单纯形网格 w∈{0,0.05,...,1}，231 组
- **前后半窗确认**：前 122 期选权重、后 121 期确认，只接受两半都优于最优单模型的组合
- 只取平台区（权重 ±0.1 扰动 IC 变化 < 0.001），不取尖峰
- 参考基线：等权 (1/3, 1/3, 1/3) 必须一起评估

### Step 5：应用到测试集 + 验收
- 用选定权重对三份 (442, 5282) 测试预测做同样的逐时点秩加权
- 输出规格与主流水线逐字节对齐：评估位 (0,1] percentile、非评估位 0.5、float32
- 验收：评估位 2,042,538 / 中性 292,106 / 全有限 / 与 y1_best.npy 的逐时点秩相关（应 0.8~0.98）

### Step 6：晋级判定
- 集成 Valid IC（后半确认窗）> 0.0929 + 0.001 才建议替换提交
- 产物：`outputs/y1_ensemble.npy` + `outputs/ensemble_report.md`（单模型复现、权重曲面、半窗确认、最终权重）

## 注意事项

- 权重拟合有一个已知偏差：LGBM 的 Valid 预测来自在 Valid 上早停的模型，其 0.0929 略偏乐观。
  严格做法是用 fold_3（valid [2674,2918)）预测拟合权重、官方 Valid 只做确认——作为对比实验一并跑
- TCN 推理时注意 `history_available_mask` 口径（486 窗口内任意一期有效），
  与评估掩码取交即可
- 所有产物放 `outputs/`，不覆盖 `y1_rank_pipeline/y1_rank_outputs/y1_best.npy`
