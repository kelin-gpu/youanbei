# 创新前验收记录

状态：`completed`（2026-08-23）。原则是复用已有证据，只补缺失测试。

## 已有测试，直接复用

- 数据与预测契约：`scripts/project_audit.py` 和 exp016 safety contract。
- 原始特征质量、异常率、Train/Valid/Test PSI：`01_analysis/outputs/numeric_feature_profile.csv`。
- 单特征 Train/Valid RankIC、std、正率：`01_analysis/outputs/rankic_summary.csv`。
- 三折 walk-forward OOF：exp016 OOF 缓存及 exp021 `metrics_fit.json`。
- exp016 全量 Valid、逐时间稳定性：`official_valid_full_report.json`、`per_time_ic_full.csv`。
- 家族消融与资源审计：`family_ablation.json`、`resource.json`。

这些测试不重新执行。

## 本次补齐

1. `drift_feature_registry.csv`：把已有 PSI、异常率和 Train/Valid RankIC 合并为统一特征登记。动作分类只使用 Train/Valid 证据，Test PSI 仅作监控。
2. `prediction_similarity.json`：exp021 与 exp023h 在 2,042,538 个评估位上的 Pearson 相关为 `0.995258`；逐截面秩相关均值 `0.994910`，只有前 6 个截面发生实质变化。
3. `exp021_validation_audit/`：复用 exp016 全量 Valid 家族缓存和 exp021 head/router，仅在内存重建 official-valid categorical tabular；不保存模型、不覆盖基线。

exp021 全量 Valid 结果：

- mean RankIC：`0.091809`
- std RankIC：`0.097927`
- 正 IC 比例：`83.13%`
- 最差截面：`-0.171552`
- 最差 10% 截面均值：`-0.092179`
- 后半段均值：`0.085609`
- 高漂移 20% 截面均值：`0.089161`
- 其余截面均值：`0.092477`
- 高漂移差值：`-0.003316`

## 验收结论

- exp021 比 exp016 同口径全量 Valid 高约 `0.000419`，与线上小幅提升方向一致。
- 高漂移时段存在可测的负向差异，但幅度有限；支持将漂移门控作为单变量 exp024 验证，不支持直接替换整个模型栈。
- exp021 预测与正式提交的 SHA-256 在测试前后完全一致。
- 本次未生成提交候选，未修改任何正式文件。
