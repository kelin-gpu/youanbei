# 模型动物园实验报告（exp_012_model_zoo）

- 目标：在"锚点 + 在任近期 1702 期专家"核心阵容之外，注册低成本正交候选动物并统一裁决。
- 数据口径：processed_data_v1；全部动物按"近期 1702 期专家"协议训练（与 exp_009/010 一致），锚点只读。
- 种子：42 / 2026 / 3407（Stage B 3 种子集成）。
- 决策记录：`04_results/_decision_log/`（逐动物）。
- 训练终点：Train-only（2918）（阶段 1 决策）。

## 1. Stage A 筛选（fold_1/fold_2，seed 42，权重网格 0.25/0.30/0.35）

| 动物 | 框架 | 特征 | 最佳Δ | fold_1 | fold_2 | 在任专家Δ(0.30) | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| lgbm_xendcg | LGBM rank_xendcg | legacy_328 | -0.002341 | -0.003441 | -0.001242 | +0.001022 | 不通过 |
| lgbm_cat337 | LGBM lambdarank | legacy_328+9类别(337) | **+0.001307** | +0.001015 | +0.001600 | +0.001022 | 通过 |
| catboost_yetirank | CatBoost YetiRank | legacy_328 | **+0.007512** | +0.007676 | +0.007349 | +0.001022 | 通过 |

结论：rank_xendcg 目标对 Y1 不适用；类别感知 337 维首次在"专家+门槛"语境显示折内增益（此前 exp_005 仅测过 419 整体视图）；CatBoost YetiRank 折内增益约为在任专家的 7 倍。

## 2. Stage B 确认（3 种子集成）

| 动物 | 影子Δ | 官方ValidΔ | Test-锚点秩相关 | 判定 |
|---|---:|---:|---:|---|
| lgbm_cat337 | +0.000719 | **+0.000754** | 0.8663 | 全门槛通过（官方 Valid 为正，优于在任 3 种子 -0.000022） |
| catboost_yetirank | **+0.004024** | **+0.004341** | 0.8798 | 折/影子/Valid 三级一致强正向 |

- 官方 Valid 明细（catboost，3 种子）：锚点 0.092749 → 融合(w=0.35) 0.097090，Δ=+0.004341。
- Test-锚点相关性偏低（0.87-0.88）为预期：模型族不同（CatBoost vs LGBM）+ 训练终点不一致（Train-only vs 官方锚点 3161）。

## 3. 结论

1. CatBoost YetiRank 是有效的第二棵 GBDT：三级验证一致强正向，是本项目首次在开发折、影子、官方 Valid 三个层面同时显著改善的候选组件。
2. 类别感知 337 视图在专家语境下有增益（官方 Valid Δ=+0.000754），可作为后续特征工厂的证据（block tree_419_categorical 的准入裁决由"待裁决"更新为"折内+Valid 双正"）。
3. xendcg 目标被淘汰，归档为已排除假设。

## 4. 线上结果回填（2026-08-07）

- `promoted_candidate.npy`（锚点 0.65 + CatBoost YetiRank 0.35，Train-only）线上 RankIC：**0.108265**（用户提供）。
- 对比：正式提交 exp_003（0.108105，+0.000160）；线上最佳 exp_007（0.109959，-0.001694）；同口径 Train-only 的 exp_011（0.104988，+0.003277）。
- 判定：**未晋级**（未超过线上最佳），正式提交目录保持原样。
- 解读：CatBoost 专家在线上确认有效（Train-only 口径内线上最高），但本地 Valid（Δ+0.0043）显著高估了绝对增益（线上仅 +0.000160 相对 exp_003），本地-线上分布差异再次得到验证。CatBoost 家族转移衰减已标定进 `y1_fw_lib.py` 的 TRANSFER_TABLE。

## 5. 产物

- `zoo_screen.csv` / `zoo_confirmation.csv`：筛选与确认明细。
- `animal_lgbm_cat337_{shadow,valid,test}.npy`、`animal_catboost_yetirank_{shadow,valid,test}.npy`：全区间分量（供阶段 4 融合）。
- `runtime_cache/`：全部动物预测按指纹缓存（含折内分量，可复现）。
- 决策记录：`20260807_*_animal_lgbm_cat337.json`、`20260807_*_animal_catboost_yetirank.json`。
