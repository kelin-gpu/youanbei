# exp027：检索增益归因与稳健收缩

状态：`completed_rejected`。

`run_exp027a.py` 在 exp024a 的相同四个严格后置窗口上，将 exp024b 的检索校正拆为全历史稳定先验与状态特异残差。默认流程不读取 Test、不生成预测，也不触发线上提交。

exp027a 实际结论为 `inconclusive_keep_exp024b`，未通过预注册门槛。因此未建立 exp027b，正式基线继续保持 exp024b `0.120847`。

```powershell
D:\anaconda\anaconda_data\envs\jingge_ts\python.exe 02_experiments\exp_027_retrieval_attribution\run_exp027a.py
```
