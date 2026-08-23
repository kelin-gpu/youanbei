# exp028a：时间有序个股残差收缩校准

状态：`completed_failed_evidence_only_prediction_generated`。

唯一变化是：在冻结基线截面秩上加入仅由更早严格 OOF 残差估计的个股经验贝叶斯收缩偏置。没有训练新模型、没有窗口内标签更新、没有未来特征或 Test 标签。

## 诊断结果

- fold2 delta：`-0.025045`；
- fold3 delta：`-0.003758`；
- official Valid delta：`-0.013392`；
- pooled delta：`-0.014209`，90% bootstrap 下界 `-0.019438`；
- 最差32截面块：`-0.059732`；
- 决策：`failed_gates_prediction_generated_evidence_only_not_promotable`。

## 强制独立预测产物

用户明确要求无论门槛结果均生成 `prediction_1.npy`。本文件 SHA-256 为 `59b4a03eff7170ad89e4556105a22650dcf6c16df630052e52a5a44475cea61d`，shape `(442, 5282)`，dtype `float32`。前 `6` 个锚点截面逐值保留 exp024b，其余截面应用冻结个股偏置。

门槛失败时该预测仅为 `evidence_only_not_promotable`；不会自动提交，也不会覆盖 `final_submission`。
