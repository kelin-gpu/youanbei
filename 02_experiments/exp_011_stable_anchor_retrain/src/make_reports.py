"""exp_011 报告生成：experiment_report.md + prediction_report.md + seed_results 回填 runtime。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

RESULT_DIR = Path(r"d:\google_dl\book\youanbei\04_results\exp_011_stable_anchor_retrain")

metrics = json.loads((RESULT_DIR / "metrics.json").read_text(encoding="utf-8"))
metadata = json.loads((RESULT_DIR / "metadata.json").read_text(encoding="utf-8"))
weight_search = pd.read_csv(RESULT_DIR / "weight_search.csv", encoding="utf-8-sig")
fold_results = pd.read_csv(RESULT_DIR / "fold_results.csv", encoding="utf-8-sig")
retrain_sim = pd.read_csv(RESULT_DIR / "retrain_simulation.csv", encoding="utf-8-sig")
ablation = pd.read_csv(RESULT_DIR / "feature_ablation.csv", encoding="utf-8-sig")
seed_results = pd.read_csv(RESULT_DIR / "seed_results.csv", encoding="utf-8-sig")

# 回填 seed_results runtime（从模型元数据）
def backfill_runtime():
    rows = []
    for _, r in seed_results.iterrows():
        meta_path = RESULT_DIR / f"model_{r['model_type']}_seed{int(r['seed'])}.metadata.json"
        rt = r.get("runtime")
        if (rt is None or (isinstance(rt, float) and pd.isna(rt))) and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rt = round(meta["training_seconds"], 2)
        rows.append({**r.to_dict(), "runtime": rt})
    return pd.DataFrame(rows)


seed_results = backfill_runtime()
seed_results.to_csv(RESULT_DIR / "seed_results.csv", index=False, encoding="utf-8-sig")

fp = metrics["final_prediction"]
fpv = metrics["final_path"]
ov_anchor = fpv["official_valid_anchor"]
ov_blend = fpv["official_valid_blend"]
sh_anchor = fpv["shadow_anchor"]
sh_blend = fpv["shadow_blend"]
sel_w = metrics["selected_weight"]
retrain_d = metrics["retrain_deltas"]

fmt6 = lambda x: f"{x:.6f}"

# =====================================================================
# experiment_report.md
# =====================================================================
abl_sum = ablation.groupby("feature_config").agg(
    feature_count=("feature_count", "first"),
    fold_1_delta=("delta_vs_full", lambda s: float(s.iloc[0])),
    fold_2_delta=("delta_vs_full", lambda s: float(s.iloc[-1])),
    mean_delta=("delta_vs_full", "mean"),
    worst_fold_delta=("delta_vs_full", "min"),
).reset_index()
abl_md = "\n".join(
    f"| {r.feature_config} | {int(r.feature_count)} | {fmt6(r.fold_1_delta)} | {fmt6(r.fold_2_delta)} | {fmt6(r.mean_delta)} | {fmt6(r.worst_fold_delta)} |"
    for r in abl_sum.itertuples()
)

ws_md = "\n".join(
    f"| {r.recent_weight} | {fmt6(r.fold_1_rankic)} | {fmt6(r.fold_2_rankic)} | {fmt6(r.mean_delta)} | {'是' if r.selected else '否'} |"
    for r in weight_search.itertuples()
)

fold1 = fold_results[fold_results.fold == "fold_1"]
fold2 = fold_results[fold_results.fold == "fold_2"]

exp_report = f"""# exp_011 稳定锚点重训实验报告

## 1. 实验目标

在 legacy_328 + LightGBM LambdaRank 稳定主线上，新增并执行一套独立、可复现、端到端的 Y1 预测实验。优先验证两个问题：

1. 当前测试预测模型的实际训练终点是否为 Train 终点 2918（审计结论：**否**，exp_003/006/007/009 的测试模型均训练至 3161，包含 Valid 标签）。
2. 参数完全冻结后，将官方 Valid 标签加入最终训练（训练到 3161），是否能够稳定改善后续时间段预测（本次结论：**不能稳定改善**，见第 7 节重训模拟）。

## 2. 数据与缓存核查

- `data.z` SHA-256 = `a426a7078097e8d970c2f27a30a49b3122a8a0ea7c4c05f35938d5f568cfd04c`，与 manifest 一致。
- `processed_data_v1` 的 READY 与 manifest SHA-256 = `f7c4076de6e3ae7d631554df5a15f69f50d7e8f676249fb6d2d4cf71ccec8c6f`，21 个缓存文件全部校验通过。
- 官方时间边界：Train [486,2918)、Valid [2918,3161)、Test [3161,3603)、T=3603、S=5282，全部匹配。
- 监督样本：Train 6,489,099 / Valid 982,972 / Test 2,042,538（mask_y），与缓存行数一致。
- legacy_328 特征视图与历史缓存最大绝对误差 < 1e-7，直接复用缓存（详见 `audit_report.md`）。

## 3. 锚点复现

固定使用 exp_003 调优参数（learning_rate=0.0228695、num_leaves=79、min_data_in_leaf=147、feature_fraction=0.80936、bagging_fraction=0.647764、lambda_l1=2.35724、lambda_l2=0.238705、max_bin=127、objective=lambdarank、8 轮）。

- 复现官方 Valid RankIC：`{fmt6(metrics['anchor_reproduction']['reproduced'])}`（期望 `{fmt6(metrics['anchor_reproduction']['expected'])}`，容差 0.0003）。
- 判定：**通过**（精确复现）。
- 开发折锚点同样精确复现 exp_009：fold_1=`{fmt6(fold1[fold1.model=='anchor'].iloc[0]['mean_rankic'])}`（exp_009: 0.112031）、fold_2=`{fmt6(fold2[fold2.model=='anchor'].iloc[0]['mean_rankic'])}`（exp_009: 0.097117）、影子验证锚点=`{fmt6(sh_anchor['mean_rankic'])}`（exp_009 fold_3: 0.083772）。

## 4. 近期 1702 期专家

- 训练区间 [1216,2918)、16 轮、其余参数与锚点一致。
- 官方 Valid RankIC：`{fmt6(metrics['official_valid_recent_s42']['mean_rankic'])}`（exp_009 记录 0.089657，一致）。
- 与锚点逐时间截面秩相关：0.8936。

## 5. 多随机种子

种子 42/2026/3407（官方端点 [486,2918) 锚点与 [1216,2918) 专家），各种子指标见 `seed_results.csv`。种子间方差显著（锚点 Valid RankIC 0.085842~0.092940），3 种子集成：

- 锚点集成 Valid RankIC `{fmt6(metrics['official_valid_anchor_ens']['mean_rankic'])}`（std {fmt6(metrics['official_valid_anchor_ens']['rankic_std'])}）。
- 专家集成 Valid RankIC `{fmt6(metrics['official_valid_recent_ens']['mean_rankic'])}`。
- 多种子稳定性判定：**不稳定**（集成官方 Valid 融合增量 {fmt6(metrics['official_valid_blend_ens']['mean_rankic'] - metrics['official_valid_anchor_ens']['mean_rankic'])} < 单种子 42 融合增量），因此最终回退到**单随机种子 42**版本。

## 6. 三级验证体系

### 6.1 开发折

| 折 | 验证区间 | 锚点 RankIC | 近期专家 RankIC |
|---|---|---|---|
| fold_1 | [2189,2432) | {fmt6(fold1[fold1.model=='anchor'].iloc[0]['mean_rankic'])} | {fmt6(fold1[fold1.model=='recent_expert'].iloc[0]['mean_rankic'])} |
| fold_2 | [2432,2675) | {fmt6(fold2[fold2.model=='anchor'].iloc[0]['mean_rankic'])} | {fmt6(fold2[fold2.model=='recent_expert'].iloc[0]['mean_rankic'])} |

权重搜索（0.25/0.30/0.35）：

| 权重 | fold_1 RankIC | fold_2 RankIC | 平均增量 | 选中 |
|---|---|---|---|---|
{ws_md}

最佳平均增量 = {fmt6(float(weight_search['mean_delta'].max()))}，95% 阈值 = {fmt6(float(weight_search['mean_delta'].max()) * 0.95)}，达到阈值的最低权重 = **{sel_w}**。

### 6.2 Train 末端影子验证 [2675,2918)

- 单种子 42：锚点 {fmt6(sh_anchor['mean_rankic'])}、融合 {fmt6(sh_blend['mean_rankic'])}、增量 **{fmt6(fpv['shadow_delta'])}**。
- 3 种子集成：锚点 {fmt6(metrics['shadow_anchor_ens']['mean_rankic'])}、融合 {fmt6(metrics['shadow_blend_ens']['mean_rankic'])}、增量 {fmt6(metrics['shadow_delta_ens'])}。

### 6.3 官方 Valid [2918,3161)（一次性检查）

- 锚点（seed42）：{fmt6(ov_anchor['mean_rankic'])}（前半 {fmt6(ov_anchor['first_half_rankic'])} / 后半 {fmt6(ov_anchor['second_half_rankic'])} / 最差季度 {fmt6(ov_anchor['worst_quarter_rankic'])}）。
- 融合（w={sel_w}，seed42）：{fmt6(ov_blend['mean_rankic'])}（前半 {fmt6(ov_blend['first_half_rankic'])} / 后半 {fmt6(ov_blend['second_half_rankic'])} / 最差季度 {fmt6(ov_blend['worst_quarter_rankic'])}）。
- 增量：**{fmt6(fpv['official_delta'])}**。

## 7. 重训模拟（加入新标签后重训是否改善下一段预测）

| 模拟 | 旧模型训练终点 | 新模型训练终点 | 预测区间 | 旧融合 RankIC | 新融合 RankIC | 增量 |
|---|---|---|---|---|---|---|
| 模拟一 | 2189 | 2432 | [2432,2675) | {fmt6(retrain_sim[(retrain_sim.simulation=='sim_1')&(retrain_sim.model=='old')].iloc[0]['blend_rankic'])} | {fmt6(retrain_sim[(retrain_sim.simulation=='sim_1')&(retrain_sim.model=='new')].iloc[0]['blend_rankic'])} | {fmt6(retrain_d['sim_1'])} |
| 模拟二 | 2432 | 2675 | [2675,2918) | {fmt6(retrain_sim[(retrain_sim.simulation=='sim_2')&(retrain_sim.model=='old')].iloc[0]['blend_rankic'])} | {fmt6(retrain_sim[(retrain_sim.simulation=='sim_2')&(retrain_sim.model=='new')].iloc[0]['blend_rankic'])} | {fmt6(retrain_d['sim_2'])} |

- 两个模拟平均增量 {fmt6(retrain_d['average'])}，但模拟一出现严重负向回撤（{fmt6(retrain_d['sim_1'])}，前半段 {fmt6(float(retrain_sim[(retrain_sim.simulation=='sim_1')&(retrain_sim.model=='delta')].iloc[0]['first_half_rankic']))}、最差季度 {fmt6(float(retrain_sim[(retrain_sim.simulation=='sim_1')&(retrain_sim.model=='delta')].iloc[0]['worst_quarter_rankic']))}），更新模型的预测未出现异常漂移但增益不稳定。
- **判定：未通过**。最终测试模型不使用 Train+Valid 重训，仍使用 Train-only（锚点 [486,2918)、近期专家 [1216,2918)）。

## 8. legacy_328 内部特征块消融

| 配置 | 特征数 | fold_1 增量 | fold_2 增量 | 平均增量 | 最差折增量 |
|---|---|---|---|---|---|
{abl_md}

- 所有消融在 fold_1 均下降，没有任何特征块满足"两个开发折均不下降且平均 RankIC 提高"，**继续使用完整 legacy_328**。

## 9. 最终测试预测

- 最终策略：**{metrics['final_strategy']}**（Train-only 锚点 + Train-only 近期专家 + 融合权重 {sel_w} + 单随机种子 42）。
- 预测文件：`prediction.npy`，shape `(442, 5282)`，dtype `float32`，评价位置 2,042,538，非评价位置 292,106 全部为 0.5，SHA-256 = `{fp['sha256']}`。
- 与正式提交锚点（exp_003）平均截面秩相关：{fmt6(fp['test_anchor_rank_correlation'])}（< 0.98 门槛，因正式锚点训练至 3161 而本实验最终模型训练至 2918）。

## 10. 晋级判定

- 状态：**{metrics['promotion_status']}**。
- 开发折平均增量 {fmt6(float(weight_search['mean_delta'].max()))} ≥ 0.0005；两折增量均正。
- 影子验证增量 {fmt6(fpv['shadow_delta'])} > 0；官方 Valid 增量 {fmt6(fpv['official_delta'])} > 0。
- 官方 Valid 后半段 {fmt6(ov_blend['second_half_rankic'])}（锚点 {fmt6(ov_anchor['second_half_rankic'])}，未下降）；最差季度 {fmt6(ov_blend['worst_quarter_rankic'])}（锚点 {fmt6(ov_anchor['worst_quarter_rankic'])}，未明显下降）。
- Test 与正式锚点平均截面秩相关 {fmt6(fp['test_anchor_rank_correlation'])} < 0.98 → **未通过**。

## 11. 可选实验（探索性）

- 分层股票抽样（fold_2 anchor）：确定性 0.097117 vs 分层 0.096326，增量 -0.000792，**无增益**，不采用（见 `stratified_sampling.csv`）。
- 小权重 TCN/线性融合（官方 Valid 探索）：四组预设权重（锚点 ≥0.80、TCN ≤0.15、线性 ≤0.05）官方 Valid 均有提升（+0.0027~+0.0054），但缺少开发折/影子验证分量（exp_004 未保存折内分量），无法满足三级验证要求，**不采用**，权重回退 TCN=0、Linear=0（见 `tiny_blend_results.csv`）。
- 市场状态相似度加权（fold_2 探索）：full 0.097117 vs 状态加权 0.097959，增量 +0.000842，单一折探索性结果，未进入最终融合（见 `state_similarity.csv`）。

## 12. 结论

- 本实验完整复现了 exp_003 锚点（0.092940）与 exp_009 的折内基线，确认了实验环境与数据口径的复现性。
- 审计确认：现有测试预测模型（exp_003/006/007/009）实际训练到 3161（含 Valid 标签），并非只训练到 2918。
- 重训模拟未通过：加入最近一段标签不能稳定改善下一段预测（模拟一严重负向），因此最终测试模型保持 Train-only（训练到 2918）。
- 多随机种子（42/2026/3407）方差显著，3 种子集成在官方 Valid 融合上不稳定，最终回退单种子 42。
- 完整生成了 `prediction.npy`、`train_only_prediction.npy`、`train_valid_retrain_prediction.npy`、`valid_prediction.npy`、`shadow_prediction.npy` 与全部指标/元数据/模型文件。
- 状态：**not_promoted**；正式提交目录 `04_results/final_submission/prediction.npy` 未被修改。

## 产物清单

- `prediction.npy` / `train_only_prediction.npy` / `train_valid_retrain_prediction.npy` / `valid_prediction.npy` / `shadow_prediction.npy`
- `metrics.json` / `metadata.json` / `audit_report.md` / `audit_stats.json` / `run.log`
- `fold_results.csv` / `weight_search.csv` / `seed_results.csv` / `retrain_simulation.csv` / `feature_ablation.csv`
- `model_anchor_seed{42,2026,3407}.txt`(+metadata) / `model_recent_seed{42,2026,3407}.txt`(+metadata)
- `stratified_sampling.csv` / `tiny_blend_results.csv` / `state_similarity.csv`
"""

atomic = RESULT_DIR / "experiment_report.md.partial"
atomic.write_text(exp_report, encoding="utf-8")
import os
os.replace(atomic, RESULT_DIR / "experiment_report.md")
print("experiment_report.md 已生成。")
