# 04 结果

状态：`archived`（历史结果）。

本目录只保存需要直接查看或提交的结果。

- `最终提交/y1_final.npy`：当前可提交预测。
- `最终提交/y1_final.metadata.json`：提交文件验收信息。
- `候选提交/y1_blend_full65_recent35.npy`：全历史65%与近期1702期专家35%的Rank融合候选，尚未提交。
- `候选提交/y1_blend_full65_recent35.metadata.json`：候选来源、Valid结果和SHA-256。
- `模型结果汇总.csv`：当前主要模型的本地 Valid 对比。

训练过程中的大模型权重和中间预测仍跟随对应训练 Notebook 保存，避免复制大文件。
