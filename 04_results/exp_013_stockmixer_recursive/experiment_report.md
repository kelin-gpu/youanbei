# exp_013 Mask-aware StockMixer-Lite + 递归预测

- 状态：完整训练与推理完成
- 模型：单一 StockMixer-Lite，seed=42，EMA=0.999
- 训练：20 epochs × 512 时间截面/epoch
- Valid 原始 RankIC：`0.087804`
- Valid 递归 RankIC：`0.087884`
- 递归增量：`+0.000080`
- Test 预测 SHA-256：`4b61d76d34a90c16b33f91f5e411618e0d8191ff16bf4bc829407d1c86238131`
- 当前线上最佳参考：`0.109959`

本实验不融合任何其他模型，也不会自动覆盖 `04_results/final_submission/prediction.npy`。
