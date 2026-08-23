import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/google_dl/book/youanbei/outputs/project_review_20260804";
const outputPath = `${outputDir}/友安杯_项目结果与代码方法对照表.xlsx`;

const COLORS = {
  navy: "#1F4E78",
  blue: "#DCE6F1",
  teal: "#0F766E",
  tealLight: "#D9EDE9",
  green: "#E2F0D9",
  greenDark: "#548235",
  amber: "#FFF2CC",
  amberDark: "#A66A00",
  red: "#FCE4D6",
  redDark: "#C00000",
  gray: "#F2F2F2",
  grayText: "#666666",
  border: "#B4C7DC",
  white: "#FFFFFF",
};

function styleTitle(sheet, title, subtitle, lastColumn) {
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, fontSize: 18 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeightPx = 38;

  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: COLORS.blue,
    font: { color: "#27408B", fontSize: 10 },
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A2:${lastColumn}2`).format.rowHeightPx = 30;
}

function styleHeader(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white, fontSize: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#9FBAD0" },
  };
  range.format.rowHeightPx = 36;
}

function styleBody(range) {
  range.format = {
    font: { fontSize: 10, color: "#1F2937" },
    verticalAlignment: "top",
    wrapText: true,
    borders: {
      insideHorizontal: { style: "thin", color: "#D9E2F3" },
      bottom: { style: "thin", color: COLORS.border },
    },
  };
}

function setColumnWidths(sheet, widths) {
  for (let i = 0; i < widths.length; i += 1) {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidthPx = widths[i];
  }
}

function addReportTable(sheet, range, name) {
  const table = sheet.tables.add(range, true, name);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
  return table;
}

const workbook = Workbook.create();

// ───────────────────────────── 总览 ─────────────────────────────
const overview = workbook.worksheets.add("总览");
overview.showGridLines = false;
styleTitle(
  overview,
  "2026 友安杯 Y1 项目：项目、结果与代码处理方法总览",
  "基于当前目录、实验 Notebook、结果元数据与诊断报告整理；生成日期：2026-08-04。",
  "H",
);

overview.getRange("A4:H4").values = [[
  "实际实验数", 8,
  "正式提交实验", "exp_003_lgbm_rank",
  "正式线上 RankIC", 0.108105,
  "exp_008 官方 Valid", 0.084385,
]];
overview.getRange("A4:H4").format = {
  fill: COLORS.gray,
  font: { bold: true, color: COLORS.navy, fontSize: 11 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "all", style: "thin", color: COLORS.border },
};
overview.getRange("A4:H4").format.rowHeightPx = 34;
overview.getRange("B4").format.numberFormat = "0";
overview.getRange("F4:H4").format.numberFormat = "0.000000";

const overviewHeaders = ["维度", "当前项目口径", "关键数值/范围", "结果或判断", "主要输入", "主要输出", "对应代码/目录", "依据文件"];
const overviewRows = [
  ["项目目标", "对 5,282 只股票在 442 个测试时点预测 Y1，并按时间截面排序评估", "Y1；Test=442×5,282", "评价核心是逐时间截面 RankIC 均值", "data.z", "prediction.npy", "02_experiments/exp_*/experiment.ipynb", "README.md；04_results/experiment_records.md"],
  ["原始数据", "Zstandard 压缩的 Pickle 字典；张量粒度为 (time_idx, stock_idx)", "3,603×5,282；99 数值 + 9 类别", "不是 CSV/Parquet 单表结构", "data.z", "num_x/cat_x/y1/y2/mask_x/mask_y", "各 Notebook 的 read_zstd_pickle/load_data_z", "phase1_diagnostic_report.txt；exp_008 输出"],
  ["时间切分", "按官方索引固定切分，不随机打乱时间", "Train [486,2918)；Valid [2918,3161)；Test [3161,3603)", "训练 2,432 期、验证 243 期、测试 442 期", "train_start_idx/valid_start_idx/test_start_idx", "各区间样本与组大小", "各实验切分单元格", "04_results/experiment_records.md"],
  ["样本口径", "exp_001~007 主要使用 mask_x & mask_y & finite(y1)；exp_008 改为 mask_y & finite(y1)，并对 X 做因果补全", "Test 评价位置 2,042,538", "存在需统一说明的掩码口径差异", "mask_x、mask_y、y1", "监督样本索引/补全状态", "phase1_diagnostic.py；exp_008 单元格 10~26", "phase1_diagnostic_report.txt；exp_008 Notebook"],
  ["缓存策略", "仅保存生成超过一小时的重型视图；复用前检查 READY、manifest 与 SHA-256", "processed_data_v1 约 34.8 GiB", "避免普通特征无节制落盘", "data.z/历史特征", "03_cache/processed_data_v1", "03_cache/README.md", "README.md；03_cache/README.md"],
  ["统一结果接口", "实验结果集中到 04_results/<experiment_id>/；正式提交与实验候选隔离", "标准预测 442×5,282、float32", "非评价位置应填 0.5；实验不得自动覆盖正式提交", "模型、验证预测、测试特征", "prediction.npy / metrics.json / metadata.json / model.* / report", "各 Notebook 保存单元格", "04_results/README.md"],
  ["当前正式结果", "LightGBM LambdaRank，328 个因果数值特征，8 轮", "Valid 0.092940；线上 0.108105", "当前线上最佳；final_submission 与 exp_003 SHA-256 完全一致", "40 原始筛选特征及派生视图", "04_results/final_submission/prediction.npy", "exp_003_lgbm_rank/experiment.ipynb", "experiment_records.md；final_submission/metadata.json"],
  ["本地最高但未晋级", "线性 0.4 + TCN 0.4 + LGBM-v2 0.2 的截面秩融合", "Valid 0.101285；线上 0.097028", "同段 Valid 搜权重且稳定性门槛失败，不替换正式提交", "三模型 Valid/Test 预测", "exp_004/prediction.npy、权重搜索表", "exp_004_model_ensemble/experiment.ipynb", "exp_004/experiment_report.md"],
  ["优先候选", "全历史 LightGBM 0.65 + 最近 1702 期专家 0.35", "Valid 0.094446；与 y1_best Test 秩相关 0.984067", "候选未提交；应继续降低近期权重并做多折验证", "两个 LambdaRank 模型", "exp_007/prediction.npy", "exp_007_recent_window_blend/experiment.ipynb", "experiment_records.md；exp_007/metrics.json"],
  ["最新实验 exp_008", "完整 X 因果补全 + 363 维特征 + decay_1200 + 32 轮 LambdaRank", "Train 内走步 0.100747；官方 Valid 0.084385", "预测已生成但低于正式模型，且统一结果文件未补齐", "filled_num_x 与全量股票", "exp_008/model.txt、prediction.npy", "exp_008_new_method/experiment.ipynb", "Notebook 已执行输出；目录验收"],
  ["环境", "environment.yml 定义 youan-y1；当前 exp_008 与诊断实际运行于 jingge_ts", "Python 3.12；LightGBM 4.7.0", "环境名称与说明需统一，避免复现实验时误用解释器", "environment.yml", "Conda 环境", "Notebook 环境检查单元格", "environment.yml；phase1_diagnostic_report.txt"],
  ["项目文档", "项目报告仍为草稿，且 README 仍写“七个实验”", "实际已有 8 个实验", "需更新项目实现成果、实验清单与 exp_008 结论", "05_docs/project_report", "DOCX/HTML/PDF", "05_docs/README.md", "README.md；05_docs/README.md"],
];
overview.getRange("A7:H7").values = [overviewHeaders];
overview.getRange(`A8:H${7 + overviewRows.length}`).values = overviewRows;
styleHeader(overview.getRange("A7:H7"));
styleBody(overview.getRange(`A8:H${7 + overviewRows.length}`));
overview.getRange(`A8:H${7 + overviewRows.length}`).format.rowHeightPx = 76;
addReportTable(overview, `A7:H${7 + overviewRows.length}`, "ProjectOverviewTable");
setColumnWidths(overview, [110, 270, 190, 250, 160, 200, 240, 250]);
overview.freezePanes.freezeRows(7);
overview.freezePanes.freezeColumns(1);

// ───────────────────────── 实验结果对照 ─────────────────────────
const experiments = workbook.worksheets.add("实验结果对照");
experiments.showGridLines = false;
styleTitle(
  experiments,
  "实验结果与处理方法对照",
  "数值列区分 Train 内筛选、官方 Valid 与线上 RankIC；空白表示当前项目没有对应结果。",
  "N",
);

const experimentHeaders = [
  "实验编号", "角色/方法", "核心数据处理", "模型与训练", "验证/选择方式",
  "Train内/筛选 RankIC", "官方 Valid RankIC", "补充稳定性指标", "线上 RankIC",
  "结果状态", "主要结果产物", "当前结论", "对应代码", "结果依据",
];
const experimentRows = [
  ["exp_001_linear_baseline", "最低线性基线", "99 个原始数值特征；仅 Train 拟合均值/标准差；删除标签、掩码或特征不可用样本；不做时序派生", "线性回归；mini-batch SGD；3 epochs；batch 16,384；MSE+L2", "固定时间 Train/Valid/Test 留出", null, 0.0896776, "—", 0.075549, "已完成；历史提交", "prediction.npy；model.npz；metrics/metadata/report", "线上较弱，仅保留为可解释最低基线", "02_experiments/exp_001_linear_baseline/experiment.ipynb", "experiment_records.md；exp_001/metrics.json"],
  ["exp_002_tcn_baseline", "序列深度基线", "99 数值特征；每样本读取过去 486 期；窗口缺失用股票历史均值；Train-only 标准化；附加有效性掩码和覆盖率", "TCN；100 输入通道；8 个因果残差块；AdamW；8 epochs；MSE", "固定时间留出；按时间分箱与股票抽样", null, 0.091556, "exp_004 重训分量 Valid 0.094233", 0.086096, "已完成；历史提交", "prediction.npy；model.pt；metrics/metadata/report", "线上优于线性，但仍弱于 LightGBM，只适合小权重差异信号", "02_experiments/exp_002_tcn_baseline/experiment.ipynb", "experiment_records.md；exp_002/metrics.json"],
  ["exp_003_lgbm_rank", "当前正式主模型", "40 个筛选原始数值 + 20 截面排名 + 80 滞后 + 180 滚动/变化 + 8 可用性/覆盖特征，共 328 维；因果趋势填补", "LightGBM LambdaRank；连续 Y1 转 64 档相关度；每时点最多 1,200 股；固定 8 轮", "三个扩展窗口走步折；官方 Valid 最终晋级判断", null, 0.092940168, "后半 Valid 0.091868", 0.108105, "正式提交；线上最佳", "prediction.npy；model.lgb.txt；metrics/metadata/report；诊断表", "本地不是最高，但线上最高，后续融合必须以它为锚点", "02_experiments/exp_003_lgbm_rank/experiment.ipynb", "experiment_records.md；exp_003/experiment_report.md"],
  ["exp_004_model_ensemble", "三模型秩融合", "线性/TCN/LGBM 分别重训并生成 Valid/Test；各时点先转百分位秩；231 组权重网格", "linear 0.40 + TCN 0.40 + LGBM-v2 0.20；LGBM-v2 固定 1,558 轮", "同一官方 Valid 搜权重并报告；设前后半与邻域稳定性门槛", null, 0.101285003, "前半 0.109618；后半 0.092883；邻域波动 0.001277", 0.097028, "完成但未晋级", "prediction.npy；valid_prediction.npy；模型与分量预测；weight_search.csv", "本地最高但选择过拟合和锚点错误，线上低于 exp_003", "02_experiments/exp_004_model_ensemble/experiment.ipynb", "exp_004/metrics.json；experiment_report.md"],
  ["exp_005_lgbm_feature_screen", "特征视图筛选", "比较 legacy_328、numeric_408、full_419；固定数据视图和 RankIC 口径", "LightGBM LambdaRank；比较 8/16/32 轮", "历史筛选表；selection_score 兼顾稳定性", 0.092940154, null, "selection_score 0.085883", null, "historical_metrics_only", "screening_results.csv；metrics/metadata/report；无 prediction.npy", "扩展至 408/419 维没有可靠增益，legacy_328 仍最稳", "02_experiments/exp_005_lgbm_feature_screen/experiment.ipynb", "exp_005/metrics.json；screening_results.csv"],
  ["exp_006_lgbm_window_screen", "训练窗口筛选", "比较最近 2,188/1,945/1,702/1,216/730 期；统一 legacy_328 特征", "LightGBM LambdaRank；每窗口比较 8/16/32 轮", "稳定性选择分数；完整、后半和最差季度综合", 0.094017759, null, "完整均值最高：recent1702/16轮 ≈0.094309；稳定选择 recent1702/8轮", null, "historical_metrics_only", "screening_results.csv；metrics/metadata/report；无 prediction.npy", "最近 1,702 期有增益，但宜与全历史模型组合", "02_experiments/exp_006_lgbm_window_screen/experiment.ipynb", "exp_006/metrics.json；screening_results.csv"],
  ["exp_007_recent_window_blend", "近期专家候选", "全历史与最近 1,702 期共用 legacy_328；各自按时点排名后融合", "两个 8 轮 LambdaRank；full 0.65 + recent 0.35", "官方 Valid 评估；未上线提交", null, 0.094445762, "后半 Valid 0.092851；与 y1_best Test 秩相关 0.984067", null, "候选未提交", "prediction.npy；expert_prediction.npy；model.txt；metrics/metadata/report", "当前优先候选，但 35% 近期权重可能偏大，应做多折减权搜索", "02_experiments/exp_007_recent_window_blend/experiment.ipynb", "experiment_records.md；exp_007/metrics.json"],
  ["exp_008_new_method", "因果补全全量特征新方法", "全 X 因果补全；短缺口趋势、长缺口衰减到市场中心、冷启动中心；99 raw + 99 rank + 60 surprise + 60 surprise-rank + 40 trend + 5 state = 363 维", "自动选择 decay_1200；手动选择 32 轮 LambdaRank；Train+Valid 重训；Test 分时点百分位排名", "3 个 Train 内 walk-forward；模型候选 8/16/32 轮及 xendcg_16；官方 Valid 安全检查", 0.100747, 0.084385, "Train 内后段 0.096624；官方 Valid 后段 0.078038；最差块 0.074565", null, "预测已生成；统一结果未补齐", "model.txt；prediction.npy；缺 metrics.json/metadata.json/report", "新方法在 Train 内更强但官方 Valid 明显回落，不应替换正式提交", "02_experiments/exp_008_new_method/experiment.ipynb", "Notebook 执行输出；04_results/exp_008_new_method"],
];
experiments.getRange("A4:N4").values = [experimentHeaders];
experiments.getRange("A5:N12").values = experimentRows;
styleHeader(experiments.getRange("A4:N4"));
styleBody(experiments.getRange("A5:N12"));
experiments.getRange("A5:N12").format.rowHeightPx = 104;
experiments.getRange("F5:G12").format.numberFormat = "0.000000";
experiments.getRange("I5:I12").format.numberFormat = "0.000000";
experiments.getRange("F5:I12").format.horizontalAlignment = "right";
experiments.getRange("A5:A12").format.font = { bold: true, color: COLORS.navy, fontSize: 10 };
addReportTable(experiments, "A4:N12", "ExperimentComparisonTable");
setColumnWidths(experiments, [185, 150, 310, 260, 220, 105, 105, 230, 100, 145, 235, 255, 260, 240]);
experiments.freezePanes.freezeRows(4);
experiments.freezePanes.freezeColumns(2);
experiments.getRange("J5:J12").conditionalFormats.add("containsText", { text: "正式", format: { fill: COLORS.green, font: { color: COLORS.greenDark, bold: true } } });
experiments.getRange("J5:J12").conditionalFormats.add("containsText", { text: "未", format: { fill: COLORS.amber, font: { color: COLORS.amberDark, bold: true } } });
experiments.getRange("J5:J12").conditionalFormats.add("containsText", { text: "historical", format: { fill: COLORS.gray, font: { color: COLORS.grayText } } });

// ───────────────────────── 代码处理链路 ─────────────────────────
const codeMap = workbook.worksheets.add("代码处理链路");
codeMap.showGridLines = false;
styleTitle(
  codeMap,
  "代码处理链路与实现位置",
  "按数据进入、特征处理、训练验证、融合和提交验收梳理主要代码方法。",
  "H",
);

const codeHeaders = ["处理环节", "模块/实验", "输入", "代码处理方法", "代表性函数/类", "输出", "保护与验收", "源文件"];
const codeRows = [
  ["目录与数据侦查", "phase1_diagnostic.py", "当前项目全部文件、data.z、Notebook/CSV/文本", "盘点文件；读取最大表格；抽样与全量缺失画像；识别数据粒度；静态审计模型、切分、泄露与预处理线索", "inventory_report；profile_raw_competition_data；audit_code；final_summary", "phase1_diagnostic_report.txt", "报告明确声明只做阶段一诊断，不训练、不生成提交", "phase1_diagnostic.py"],
  ["独立数据分析", "01_analysis", "data.z", "分析数据结构、股票池、掩码、标签分布、特征质量、漂移、单特征 RankIC 与类别特征", "Notebook 分析单元格", "图表与分析结论（不参与训练）", "与实验训练解耦，避免把探索步骤当作必须流水线", "01_analysis/data_analysis.ipynb"],
  ["原始数据读取", "exp_001~008", "data.z", "读取压缩字节 → Zstandard 解压 → Pickle 反序列化；exp_003/004 另有 LimitedReader 与临时未压缩缓存", "read_zstd_pickle；load_data_z；ensure_uncompressed_pickle；CompetitionData", "num_x/cat_x/y1/y2/mask_x/mask_y 与官方索引", "检查路径、顶层键、shape、dtype 与切分索引", "02_experiments/*/experiment.ipynb"],
  ["线性特征准备", "exp_001", "99 个原始数值特征、掩码、y1", "按时点提取有效样本；只用 Train 分块计算均值/标准差；方差下限保护；小批量标准化", "get_labeled_chunk；summarize_logical_missing；count_nonfinite", "标准化训练批与验证/Test 特征", "NaN/Inf、重复行、维度和缺失统计", "02_experiments/exp_001_linear_baseline/experiment.ipynb"],
  ["线性训练与预测", "exp_001", "标准化特征与 y1", "Mini-batch SGD 更新权重和偏置；逐时点预测；计算逐时点 Spearman/RankIC；保存模型与预测", "update_one_batch；train_one_epoch；predict_time_point；calculate_rank_ic", "model.npz；prediction.npy；metrics/metadata/report", "SHA-256、shape、dtype、有限性检查", "02_experiments/exp_001_linear_baseline/experiment.ipynb"],
  ["TCN 窗口生成", "exp_002", "每只股票过去 486 期数值序列", "计算窗口边界；按股票历史窗口均值填缺失；Train-only 标准化；增加有效性掩码通道与历史覆盖率", "window_bounds；prepare_window_arrays；build_model_batch；history_available_mask", "100 通道序列张量与覆盖率输入", "动态滑窗、无未来信息、有限性检查", "02_experiments/exp_002_tcn_baseline/experiment.ipynb"],
  ["TCN 训练", "exp_002", "486 期序列张量", "8 个 dilation=1~128 的因果残差块；GroupNorm/GELU/dropout；AdamW + 余弦调度 + 梯度裁剪", "CausalResidualBlock；WindowTCN；evaluate_sampled_validation", "model.pt；prediction.npy；验证 RankIC", "按时间分箱抽样；恢复最佳模型；提交前检查", "02_experiments/exp_002_tcn_baseline/experiment.ipynb"],
  ["LambdaRank 特征工程", "exp_003", "原始数值、历史缓存、掩码、类别数据", "筛 40 个数值；构造截面排名、1/5/20/60 滞后、5/20/60 滚动均值/标准差/变化、覆盖与存续特征；因果缺失处理", "discover_numeric_features；build_history_cache；build_feature_matrix；_trend_estimate", "328 维纯数值特征矩阵", "人工遮挡重建、特征阶段消融、类别/交叉项诊断", "02_experiments/exp_003_lgbm_rank/experiment.ipynb"],
  ["LambdaRank 训练与调参", "exp_003", "328 维特征、连续 y1、时间组", "每时点把 y1 转 64 档相关度；确定性股票抽样；Optuna/参数验证；走步折 RankIC；官方 Valid 晋级", "train_ranker；train_ranker_no_validation；tune_parameters；verify_parameter_sets；run_pipeline", "model.lgb.txt；prediction.npy；完整实验报告", "走步验证、官方 Valid 只做最终判断、独立 acceptance_validation", "02_experiments/exp_003_lgbm_rank/experiment.ipynb"],
  ["三模型秩融合", "exp_004", "线性/TCN/LGBM 的 Valid/Test 原始预测", "逐时间截面转百分位秩；0.05 步长搜索 231 组权重；检查前后半分数、邻域稳定与主模型偏离", "cross_sectional_percentile_rank；rank_ic_by_time；权重搜索函数", "融合预测与权重明细", "多门槛 promoted=False 时不覆盖 final_submission", "02_experiments/exp_004_model_ensemble/experiment.ipynb"],
  ["特征视图筛选", "exp_005", "processed_data_v1 的 328/408/419 维固定视图", "验证 READY/manifest/SHA；统一切片和组大小；固定 LightGBM 参数比较特征集与轮数；截面秩诊断", "load_common；load_tree；train_screen_config；score_prediction", "screening_results.csv；best_key", "历史仅保留指标；新 Notebook 可重新生成完整预测", "02_experiments/exp_005_lgbm_feature_screen/experiment.ipynb"],
  ["训练窗口筛选", "exp_006", "legacy_328 与多个训练起点", "保持特征/模型不变，只改变训练历史长度；比较 8/16/32 轮；用完整、后半、最差季度组成选择分数", "row_slice_for_times；train_screen_config；score_prediction", "window screening_results.csv", "选择 stable recent1702/8，而非只取完整均值最高项", "02_experiments/exp_006_lgbm_window_screen/experiment.ipynb"],
  ["近期专家融合", "exp_007", "全历史与 recent1702 两个 LambdaRank 预测", "两个模型分别训练；逐时间排名；按 0.65/0.35 融合；比较 Valid 与 Test 对 y1_best 的秩相关", "capped_indices_for_split；group_rank_transform；save_atomic_npy", "prediction.npy；expert_prediction.npy；model.txt", "候选不自动写入 final_submission", "02_experiments/exp_007_recent_window_blend/experiment.ipynb"],
  ["完整 X 因果补全", "exp_008", "num_x、mask_x、mask_y 与同截面真实股票", "真实值保留；≤30 期内部短缺口用历史趋势；长缺口向当期市场中位数衰减；首次观测前用冷启动市场中心；记录来源与可信度", "GapImputationConfig；FilledNumericPanel；_fill_numeric_panel_arrays；ensure_filled_numeric_panel", "filled_num_x；fill_source；fill_confidence", "未来扰动不影响过去、全部有限、人工挖洞和指纹自检", "02_experiments/exp_008_new_method/experiment.ipynb"],
  ["363 维因果特征", "exp_008", "filled_num_x 与历史 EWMA 状态", "99 raw + 99 截面 rank + 60 surprise + 60 surprise-rank + 40 trend + 5 history_state；动态特征只在早期 Train 发现", "discover_stable_numeric_features；signed_rank_columns；iter_causal_numeric_features", "363 维 memmap 特征缓存", "特征块布局、范围、全量股票、有限性、未来独立性自检", "02_experiments/exp_008_new_method/experiment.ipynb"],
  ["训练历史策略", "exp_008", "Train 内三折 walk-forward 特征缓存", "比较 full_equal、decay_1200、decay_730、recent1702、recent30；至少 2 折优于基准且最差退化受限，否则回退", "TrainingPolicy；build_walk_forward_folds；run_training_policy_selection", "ACTIVE_PIPELINE_CONFIG=decay_1200", "每时点等总权重、时间衰减方向、选择与回退逻辑自检", "02_experiments/exp_008_new_method/experiment.ipynb"],
  ["模型选择与安全检查", "exp_008", "固定 363 维特征与 decay_1200", "比较 LambdaRank 8/16/32 轮和 xendcg_16；验证表只给建议；手动唯一选择 32 轮；完整 Train 在官方 Valid 安全检查", "ModelCandidate；run_model_candidate_validation；resolve_manual_model_selection；run_official_valid_safety_check", "32 轮模型；官方 Valid 0.084385", "手动选择唯一性自检；官方 Valid 警告不自动改模型", "02_experiments/exp_008_new_method/experiment.ipynb"],
  ["最终重训与提交", "exp_008", "Train+Valid、Test 特征缓存、32 轮 LambdaRank", "按选中 decay 策略重训一次；原始 Test 分数逐时点转 [0,1] 百分位；非评价位置填 0.5；Windows 中文路径下原子保存模型文本", "run_final_refit_and_predict；build_submission_matrix；validate_submission_matrix；save_model_text_atomic", "model.txt；prediction.npy", "shape/dtype/有限性/取值范围/非评价中性值/np.load 复读验收", "02_experiments/exp_008_new_method/experiment.ipynb"],
  ["正式提交隔离", "04_results/final_submission", "已验收 exp_003 预测", "复制并记录来源、目标、哈希；任何实验都不得自动覆盖", "文件与 metadata.json 约束", "正式 prediction.npy", "exp_003 与 final_submission SHA-256 均为 9d3224…38e6", "04_results/final_submission/README.md；metadata.json"],
];
codeMap.getRange("A4:H4").values = [codeHeaders];
codeMap.getRange(`A5:H${4 + codeRows.length}`).values = codeRows;
styleHeader(codeMap.getRange("A4:H4"));
styleBody(codeMap.getRange(`A5:H${4 + codeRows.length}`));
codeMap.getRange(`A5:H${4 + codeRows.length}`).format.rowHeightPx = 94;
codeMap.getRange(`A5:A${4 + codeRows.length}`).format.font = { bold: true, color: COLORS.navy, fontSize: 10 };
addReportTable(codeMap, `A4:H${4 + codeRows.length}`, "CodeProcessingMapTable");
setColumnWidths(codeMap, [140, 180, 220, 340, 285, 220, 250, 285]);
codeMap.freezePanes.freezeRows(4);
codeMap.freezePanes.freezeColumns(2);

// ───────────────────────── 结果文件验收 ─────────────────────────
const resultQa = workbook.worksheets.add("结果文件验收");
resultQa.showGridLines = false;
styleTitle(
  resultQa,
  "当前 prediction.npy 文件验收",
  "对现有结果文件进行 mmap 读取、维度/类型/有限性/值域与 SHA-256 检查；exp_005/006 当前没有预测文件。",
  "L",
);

const qaHeaders = ["实验/目录", "文件状态", "shape", "dtype", "大小(MB)", "最小值", "最大值", "均值", "全有限", "SHA-256", "与正式提交关系", "备注"];
const qaRows = [
  ["exp_001_linear_baseline", "存在", "442×5,282", "float32", 8.906, -0.502540648, 1.38173914, 0.504936695, true, "57da1582c4b8f9aed8bea1012911af3d4e4b88ae6303e8a2fcea9baf68e48592", "不同", "值域超出 [0,1]；如复用提交应确认是否需要截面排名/归一化"],
  ["exp_002_tcn_baseline", "存在", "442×5,282", "float32", 8.906, 0.005081177, 0.761230469, 0.509933531, true, "7b9cd7e831e3ae9a5374f35d870e5281d696ead4b1e66b1efc02af2ea71ec4e8", "不同", "历史 TCN 预测"],
  ["exp_003_lgbm_rank", "存在", "442×5,282", "float32", 8.906, 0.000629195, 1.0, 0.5, true, "9d322401a2d8fedd38dea66b97578873e721f03eeb93575dbc8bdc2a1aef38e6", "完全相同", "当前正式提交来源"],
  ["exp_004_model_ensemble", "存在", "442×5,282", "float32", 8.906, 0.000206526, 0.999461532, 0.500094652, true, "8e9f8439fb92ed032771cd2f1d2cdc2390b70f13ff59c45332e8b801b4396f4c", "不同", "结果存在但 promoted=false"],
  ["exp_005_lgbm_feature_screen", "缺失", "—", "—", null, null, null, null, null, "—", "—", "historical_metrics_only；需运行新 Notebook 才生成"],
  ["exp_006_lgbm_window_screen", "缺失", "—", "—", null, null, null, null, null, "—", "—", "historical_metrics_only；需运行新 Notebook 才生成"],
  ["exp_007_recent_window_blend", "存在", "442×5,282", "float32", 8.906, 0.000524109, 1.0, 0.500094652, true, "fb7cf4518bfd2d43196609798652a15701670f130c51166a553e6bedbbdec9ae", "不同", "候选未提交"],
  ["exp_008_new_method", "存在", "442×5,282", "float32", 8.906, 0.0, 1.0, 0.5, true, "22359cc5eb97aa81ddaf369350c07c92063e97eab4aa134c572019e107512ec5", "不同", "模型与预测已生成；缺统一 metrics/metadata/report"],
  ["final_submission", "存在", "442×5,282", "float32", 8.906, 0.000629195, 1.0, 0.5, true, "9d322401a2d8fedd38dea66b97578873e721f03eeb93575dbc8bdc2a1aef38e6", "基准", "official_final；非评价位置 292,106 个，统一为 0.5"],
];
resultQa.getRange("A4:L4").values = [qaHeaders];
resultQa.getRange("A5:L13").values = qaRows;
styleHeader(resultQa.getRange("A4:L4"));
styleBody(resultQa.getRange("A5:L13"));
resultQa.getRange("A5:L13").format.rowHeightPx = 66;
resultQa.getRange("E5:I13").format.numberFormat = "0.000000";
resultQa.getRange("J5:J13").format.font = { fontSize: 8, color: "#374151" };
addReportTable(resultQa, "A4:L13", "PredictionQaTable");
setColumnWidths(resultQa, [185, 90, 105, 85, 90, 105, 105, 105, 80, 350, 130, 300]);
resultQa.freezePanes.freezeRows(4);
resultQa.freezePanes.freezeColumns(2);
resultQa.getRange("B5:B13").conditionalFormats.add("containsText", { text: "存在", format: { fill: COLORS.green, font: { color: COLORS.greenDark, bold: true } } });
resultQa.getRange("B5:B13").conditionalFormats.add("containsText", { text: "缺失", format: { fill: COLORS.red, font: { color: COLORS.redDark, bold: true } } });

// ───────────────────────── 风险与建议 ─────────────────────────
const risks = workbook.worksheets.add("风险与建议");
risks.showGridLines = false;
styleTitle(
  risks,
  "项目一致性、结果风险与建议",
  "优先级 P0 直接影响正式结果；P1 影响复现与决策；P2 为文档、产物和质量改进。",
  "G",
);

const riskHeaders = ["优先级", "发现", "证据", "影响", "建议", "状态", "来源"];
const riskRows = [
  ["P0", "保持 exp_003 为正式提交", "exp_003 官方 Valid 0.092940、线上 0.108105；final_submission 与 exp_003 SHA-256 完全一致", "替换主模型会放弃当前线上最佳结果", "继续锁定 final_submission；所有新实验只写自身结果目录", "已确认", "experiment_records.md；final_submission/metadata.json"],
  ["P0", "exp_008 不应直接晋级", "Train 内走步 0.100747，但官方 Valid 仅 0.084385，比 exp_003 低约 0.008555", "Train 内选择优势没有迁移到官方 Valid，可能发生分布漂移或特征/补全过拟合", "先做特征块消融、补全口径对照和多折外推；未经线上小权重测试不替换主模型", "待实验", "exp_008 Notebook 输出；exp_003/metrics.json"],
  ["P1", "exp_008 统一结果接口不完整", "目录只有 model.txt 与 prediction.npy，没有 metrics.json、metadata.json、experiment_report.md", "难以自动对比、追溯配置和复现实验", "补写官方 Valid、Train 内折结果、选中策略、模型轮数、预测哈希和中文结论", "待补齐", "04_results/exp_008_new_method"],
  ["P1", "exp_008 手动选择备注与代码不一致", "实际唯一选择为 lambdarank_32，但备注写“默认选择…16轮”", "后续人员可能误判最终模型容量", "修正 MODEL_SELECTION_NOTE，并在 metadata 中写入实际候选名与轮数", "待修正", "exp_008 Notebook 单元格 64"],
  ["P1", "README 实验数量落后", "README 写“七个端到端实验”，实际存在 exp_001~exp_008 共 8 个", "项目入口说明与目录不一致", "更新 README 的结构、实验清单、当前结果和推荐运行方式", "待更新", "README.md；02_experiments"],
  ["P1", "Conda 环境名称不一致", "environment.yml 定义 youan-y1；exp_008 与诊断输出显示 jingge_ts", "复现时可能装错依赖或使用错误内核", "确定唯一推荐环境；统一 environment.yml、README、Notebook 内核元数据与环境断言", "待统一", "environment.yml；phase1_diagnostic_report.txt；exp_008 输出"],
  ["P1", "监督样本 mask 口径冲突", "诊断报告要求 mask_x & mask_y & finite(y1)；exp_008 使用 mask_y & finite(y1) 并补全 mask_x=False 的 X", "两条主线样本数量和分布不同，成绩不可直接归因于特征变化", "把“原始有效样本”与“补全后可训练样本”设为显式实验因子，做同特征同模型对照并更新文档", "待复核", "phase1_diagnostic_report.txt；exp_008 Notebook"],
  ["P1", "exp_004 存在权重选择过拟合", "同一段 Valid 搜权重和报告；邻域波动 0.001277 未过门槛；线上 0.097028 低于 exp_003", "本地最高 0.101285 不能代表线上更优", "用 y1_best 作 ≥0.70 锚点，限制线性/TCN 权重，并只接受多走步折共同提升", "已识别", "experiment_records.md；exp_004/experiment_report.md"],
  ["P2", "exp_005/006 只有历史筛选指标", "metadata 状态 historical_metrics_only，目录无 prediction.npy", "无法直接做 Test 相关性、融合或提交验收", "若仍需比较，运行新 Notebook 生成标准产物；否则明确归档为只读历史", "待决定", "exp_005/metadata.json；exp_006/metadata.json"],
  ["P2", "exp_001 预测值域超出 [0,1]", "prediction.npy 最小 -0.502541、最大 1.381739，虽 shape/dtype/finite 合格", "若后续复用或参与融合，可能与其他已排名预测尺度不一致", "进入融合前始终做逐截面百分位秩；若比赛接口要求 [0,1]，提交前显式转换", "需确认", "04_results/exp_001_linear_baseline/prediction.npy"],
  ["P2", "项目报告尚未同步最新结果", "05_docs/README.md 明确项目报告仍为草稿，需更新成果章节", "对外材料可能仍引用旧实验结构和成绩", "补入 exp_007/008、正式线上 0.108105、未晋级原因与下一轮计划", "待更新", "05_docs/README.md；05_docs/project_report"],
  ["P2", "重型缓存必须继续做完整性校验", "processed_data_v1 约 34.8 GiB；项目规定 READY、manifest 与 SHA-256 同时通过", "缓存损坏或错版会让多个实验共享错误特征", "保持指纹为缓存主键；任何特征配置变化都拒绝静默复用", "已建立规则", "03_cache/README.md；processed_data_v1/manifest.json"],
];
risks.getRange("A4:G4").values = [riskHeaders];
risks.getRange("A5:G16").values = riskRows;
styleHeader(risks.getRange("A4:G4"));
styleBody(risks.getRange("A5:G16"));
risks.getRange("A5:G16").format.rowHeightPx = 92;
risks.getRange("A5:A16").format.horizontalAlignment = "center";
risks.getRange("A5:A16").format.font = { bold: true, fontSize: 11 };
addReportTable(risks, "A4:G16", "RiskRecommendationTable");
setColumnWidths(risks, [75, 230, 330, 280, 340, 110, 280]);
risks.freezePanes.freezeRows(4);
risks.freezePanes.freezeColumns(2);
risks.getRange("A5:A16").conditionalFormats.add("containsText", { text: "P0", format: { fill: COLORS.red, font: { color: COLORS.redDark, bold: true } } });
risks.getRange("A5:A16").conditionalFormats.add("containsText", { text: "P1", format: { fill: COLORS.amber, font: { color: COLORS.amberDark, bold: true } } });
risks.getRange("A5:A16").conditionalFormats.add("containsText", { text: "P2", format: { fill: COLORS.gray, font: { color: COLORS.grayText, bold: true } } });

await fs.mkdir(outputDir, { recursive: true });

const inspections = [];
inspections.push(await workbook.inspect({
  kind: "table",
  range: "总览!A1:H19",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
  maxChars: 5000,
}));
inspections.push(await workbook.inspect({
  kind: "table",
  range: "实验结果对照!A1:N12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 14,
  maxChars: 6500,
}));
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 3000,
});

for (const [sheetName, range, filename] of [
  ["总览", "A1:H19", "preview_overview.png"],
  ["实验结果对照", "A1:N12", "preview_experiments.png"],
  ["代码处理链路", `A1:H${4 + codeRows.length}`, "preview_code_map.png"],
  ["结果文件验收", "A1:L13", "preview_result_qa.png"],
  ["风险与建议", "A1:G16", "preview_risks.png"],
]) {
  const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/${filename}`, new Uint8Array(await blob.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  inspections: inspections.map((x) => x.ndjson),
  errors: errors.ndjson,
}, null, 2));
