# exp025a：直接 RankIC 目标诊断

状态：`completed_rejected`。

本阶段不训练股票特征模型。它只在已有严格 OOF 七家族预测上，用冻结的 10% 收缩幅度拟合全局非负权重，并检查直接优化真实逐截面 Spearman 能否迁移到两个后置窗口和 official Valid。

不读取 Test 数组，不生成预测文件，不使用线上提交额度。

## 结果

- fold2 delta：`-0.000112`；
- fold3 delta：`+0.000061`；
- official Valid delta：`-0.000027`；
- 两次 expanding fit 的权重 L1 差异：`0.183803`（稳定性门槛内，但收益不迁移）；
- 三个受保护预测哈希不变。

结论：直接 Spearman 权重没有跨时间稳定增益，停止 soft-rank 校准器路线。优化器在 fold1 达到预注册 `maxfev=240` 后停止，但后续 fold 和 official Valid 的 no-go 已足够明确，不通过增加搜索预算追逐训练段。
