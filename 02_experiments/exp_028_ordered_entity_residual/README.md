# exp028：时间有序个股残差收缩校准

运行入口：`run_exp028a.py`。

脚本复用已有严格 OOF、冻结 official Valid 与 exp024b 预测，不重训模型。执行参数和门槛只从 `04_results/exp_028a_ordered_entity_residual/protocol.json` 读取。

按用户的显式产物要求，脚本无论诊断门槛是否通过都会在自身结果目录生成 `prediction_1.npy`；失败结果只可作为证据产物，不允许自动晋级、线上提交或覆盖正式提交。
