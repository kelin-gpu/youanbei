# exp027a：exp024b 增益归因

状态：`completed_rejected`。

本实验只改变历史指纹的加权方式，在 exp024a 的原始四窗口上比较 retrieval、global、recent、state residual 和32组固定随机对照。它不读取 Test、不生成 prediction、不使用线上额度。

## 结果

- retrieval 逐截面结果对 exp024a 复现误差：`0.0`；
- 指纹分解最大误差：`1.39e-17`；
- global 相对 retrieval：fold1 `+0.002833`、fold2 `-0.000332`、fold3 `-0.000147`、official Valid `+0.001089`；
- pooled global 优势：`+0.000827`，低于 `+0.001` 门槛；
- global 最差32截面时间块：`-0.004535`；
- global 在四个窗口均未超过随机对照95%分位；
- residual-only 为正窗口：`0/4`；
- 保护哈希、因果截断与不读取 Test 检查全部通过。

结论：`inconclusive_keep_exp024b`。不建立 exp027b、不生成 Test prediction、不使用线上提交次数。
