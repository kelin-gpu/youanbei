# Y1 Pipeline v2：当前状态与执行说明

状态：`archived`（历史流程）。

更新日期：2026-07-28

## 当前结论

当前已知可提交结果仍是：

`../y1_rank_pipeline/y1_rank_outputs/y1_best.npy`

它对应 LightGBM LambdaRank，本地官方 Valid RankIC 为 `0.092940`。
本目录不会覆盖该文件。

`03_ensemble.ipynb` 现在是 v2 的唯一执行入口，完整包含线性、TCN、
LightGBM 三模型从头训练、Valid/Test 推理、截面秩、权重搜索、半窗确认和
候选文件生成。交付时没有执行正式训练，因此目前还没有新的 Valid 权重、
`y1_ensemble.npy` 或可替换当前最佳的结论。

## 唯一执行入口

从本目录启动 Jupyter，并使用 `jingge_ts` 内核：

```powershell
cd 03_模型训练/y1_pipeline_v2
jupyter notebook 03_ensemble.ipynb
```

Notebook 顶部的 `FORCE_RETRAIN` 控制断点续跑：

- `False`（默认）：模型和预测产物存在且验收通过时加载，否则训练或推理。
- `True`：强制重新训练三个模型，并重新生成对应 Valid/Test 预测。

首次完整运行时，因为新产物尚不存在，三个模型都会各训练一次。

## 固定实验口径

| 阶段 | 时间范围 | 用途 |
|---|---:|---|
| Train | `[486, 2918)` | 三模型训练与 Train-only 统计 |
| Valid | `[2918, 3161)` | 单模型评估与集成权重选择 |
| Test | `[3161, 3603)` | 使用同一批已训练模型生成结果 |

- 线性模型：Mini-batch SGD，固定 3 epochs。
- TCN：486 期窗口，固定 8 epochs，不使用官方 Valid 早停或选 checkpoint。
- LightGBM：40 个数值特征、20 个历史特征、趋势填充、328 维 numeric
  前缀，固定 1558 轮，不重新早停。
- 三个模型都只训练一次；同一模型同时预测 Valid 和 Test。
- Valid 不重新并入训练。

## 集成与晋级规则

每个时点在官方掩码内转换为 `(rank - 1) / (n - 1)`，并列值使用平均秩，
掩码外固定为 `0.5`。权重网格步长为 `0.05`，共 231 组。

候选必须同时满足：

1. 前 122 期超过该半窗最佳单模型；
2. 后 121 期超过该半窗最佳单模型；
3. 后半窗 RankIC 高于 `0.093940`；
4. 可行的 `±0.1` 邻近权重，完整 Valid RankIC 最大波动小于 `0.001`。

通过时才写入 `outputs/y1_ensemble.npy`；否则只写
`outputs/ensemble_report.md` 和 `outputs/ensemble_weight_search.csv`。

最终候选验收固定包含：

- 形状 `(442, 5282)`；
- dtype `float32`；
- 全部为有限值；
- 官方评估位数量 `2,042,538`；
- 掩码外全部严格等于 `0.5`。

## 文件职责

| 文件或目录 | 职责 | 状态 |
|---|---|---|
| `03_ensemble.ipynb` | 三模型全量重训与秩集成 | 唯一执行入口 |
| `03_ensemble_plan.md` | 早期技术方案和决策记录 | 历史说明，保留 |
| `y1_pipeline_v2.ipynb` | 数据改进、LSTM 堆叠研究 | 研究记录，非当前入口 |
| `outputs/` | v2 模型、预测、权重和报告 | 不覆盖旧最佳 |
| `.cache_dp_v2/` | mmap 数据和历史特征缓存 | 复用，避免重复计算 |
| `.cache_lstm_stack/` | 已暂停 LSTM 支线的历史缓存 | 暂时保留 |

旧的 `ensemble_utils.py` 已删除，其功能已全部内嵌到
`03_ensemble.ipynb`。`smoke_v2.py` 也已删除，因为它会顺序触发训练单元，
不符合短时验证约束。

## 本次代码验证状态

本次只完成以下短时检查，没有执行正式训练：

- Notebook JSON、单元顺序、代码编译和 `jingge_ts` 内核配置；
- 合成数据下的排名、tie、空/单点掩码、权重和、RankIC、dtype 与中性值；
- 线性模型微型前向/损失测试；
- TCN 极小张量前向/损失测试；
- LightGBM 两组微型 query 的 1 轮功能链路；
- Train/Valid/Test 边界和训练函数的静态泄漏检查。

LSTM 堆叠支线继续暂停，不属于当前执行方案。
