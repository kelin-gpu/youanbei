// 友安杯 Y1 数据分析报告 · 图表渲染
// 数据来源：assets/report-data.js 中的 window.REPORT_DATA（由 analysis_results.json 生成）
(function () {
  var D = window.REPORT_DATA;
  if (!D) { return; }

  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  var SPLIT_RANGES = [
    { name: '预训练期', start: 0, stop: 486 },
    { name: '训练期', start: 486, stop: 2918 },
    { name: '验证期', start: 2918, stop: 3161 },
    { name: '测试期', start: 3161, stop: 3603 }
  ];
  var SPLIT_BAND = ['#e3e9f2', '#e8efff', '#fdeed6', '#edf4dd'];

  function axisBase(isX) {
    return {
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: isX ? { show: false } : { lineStyle: { color: rule, type: 'dashed' } }
    };
  }

  function commonTooltip() {
    return { trigger: 'axis', appendToBody: true, backgroundColor: bg2, borderColor: rule, textStyle: { color: ink } };
  }

  function addSplitBands(option) {
    option.series = option.series || [];
    var marks = [];
    SPLIT_RANGES.forEach(function (s, i) {
      marks.push({ xAxis: (s.start + s.stop) / 2, label: { show: false } });
      marks.push({ xAxis: s.start, label: { show: false } });
    });
    // 阶段背景带使用 markArea 与 series 对应，这里用辅助系列避免复杂度，改为在坐标轴用 markLine 标注边界
    option.xAxis.markLine = {
      silent: true,
      symbol: 'none',
      lineStyle: { color: ink, type: 'solid', width: 1, opacity: 0.5 },
      data: [486, 2918, 3161].map(function (v) { return { xAxis: v }; })
    };
    // 阶段标签：通过第二个辅助 series 无法共用 tooltip，改用 graphic 文本放到底部
    option.graphic = SPLIT_RANGES.map(function (s) {
      return {
        type: 'text',
        left: (s.start + s.stop) / 2 + '%',
        top: '92%',
        style: { text: s.name, fill: muted, fontSize: 11, fontFamily: 'Instrument Sans, sans-serif', textAlign: 'center' }
      };
    });
  }

  // 覆盖趋势需要按百分比坐标放置阶段标签；此处使用 0-100% 的 x 轴对应时点，单独处理
  function addSplitBandsPct(option, timeArr) {
    var total = timeArr[timeArr.length - 1] - timeArr[0];
    option.graphic = SPLIT_RANGES.map(function (s) {
      var mid = (s.start + s.stop) / 2;
      return {
        type: 'text',
        left: (mid / total) * 100 + '%',
        top: '90%',
        style: { text: s.name, fill: muted, fontSize: 11, fontFamily: 'Instrument Sans, sans-serif', textAlign: 'center' }
      };
    });
  }

  function init(id) {
    var el = document.getElementById(id);
    if (!el) { return null; }
    var chart = echarts.init(el, null, { renderer: 'svg' });
    window.addEventListener('resize', function () { chart.resize(); });
    return chart;
  }

  // ---------- 图 2-1 覆盖趋势 ----------
  var chartCover = init('chart-cover-trend');
  if (chartCover) {
    var bt = D.coverage.by_time;
    var optCover = {
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis' }),
      legend: { data: ['有效特征 mask_x', '标签池 mask_y', 'Y1 有限标签', 'Y1 可用标签'], textStyle: { color: muted }, top: 8 },
      grid: { left: 56, right: 20, top: 46, bottom: 42 },
      xAxis: Object.assign(axisBase(true), { type: 'category', boundaryGap: false, data: bt.time, axisLabel: { color: muted, fontSize: 10 } }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: '股票数量', nameTextStyle: { color: muted }, splitLine: { lineStyle: { color: rule, type: 'dashed' } } }),
      series: [
        { name: '有效特征 mask_x', type: 'line', data: bt.mask_x_count, smooth: true, showSymbol: false, lineStyle: { width: 2, color: accent }, itemStyle: { color: accent }, emphasis: { focus: 'series' } },
        { name: '标签池 mask_y', type: 'line', data: bt.mask_y_count, smooth: true, showSymbol: false, lineStyle: { width: 1.6, color: accent2 }, itemStyle: { color: accent2 }, emphasis: { focus: 'series' } },
        { name: 'Y1 有限标签', type: 'line', data: bt.finite_y1_count, smooth: true, showSymbol: false, lineStyle: { width: 1.2, color: muted, type: 'dashed' }, itemStyle: { color: muted } },
        { name: 'Y1 可用标签', type: 'line', data: bt.usable_y1_count, smooth: true, showSymbol: false, lineStyle: { width: 1.6, color: '#8a6d1a' }, itemStyle: { color: '#8a6d1a' } }
      ]
    };
    optCover.xAxis.markLine = {
      silent: true, symbol: 'none',
      lineStyle: { color: ink, type: 'solid', width: 1, opacity: 0.45 },
      data: [{ xAxis: 486 }, { xAxis: 2918 }, { xAxis: 3161 }]
    };
    addSplitBandsPct(optCover, bt.time);
    chartCover.setOption(optCover);
  }

  // ---------- 图 2-2 分阶段覆盖率 ----------
  var chartSplit = init('chart-cover-split');
  if (chartSplit) {
    var bySplit = D.coverage.by_split;
    var splitNames = ['pretrain', 'train', 'valid', 'test'];
    var splitLabels = { pretrain: '预训练期', train: '训练期', valid: '验证期', test: '测试期' };
    var optSplit = {
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis', valueFormatter: function (v) { return (v * 100).toFixed(2) + '%'; } }),
      legend: { data: ['有效特征 mask_x', '标签池 mask_y', '可用标签'], textStyle: { color: muted }, top: 8 },
      grid: { left: 56, right: 20, top: 46, bottom: 30 },
      xAxis: Object.assign(axisBase(true), { type: 'category', data: splitNames.map(function (n) { return splitLabels[n]; }) }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: '占比', nameTextStyle: { color: muted }, axisLabel: { color: muted, formatter: function (v) { return (v * 100).toFixed(0) + '%'; } } }),
      series: [
        { name: '有效特征 mask_x', type: 'bar', data: bySplit.map(function (r) { return r.mask_x_true_rate; }), itemStyle: { color: accent }, barGap: '20%' },
        { name: '标签池 mask_y', type: 'bar', data: bySplit.map(function (r) { return r.mask_y_true_rate; }), itemStyle: { color: accent2 } },
        { name: '可用标签', type: 'bar', data: bySplit.map(function (r) { return r.usable_y1_rate; }), itemStyle: { color: '#c9a227' } }
      ]
    };
    chartSplit.setOption(optSplit);
  }

  // ---------- 图 3-1 标签直方图 ----------
  var chartHist = init('chart-label-hist');
  if (chartHist) {
    var edges = D.label.histogram.train.edges;
    var midPoints = [];
    for (var i = 0; i < edges.length - 1; i++) { midPoints.push(((edges[i] + edges[i + 1]) / 2).toFixed(3)); }
    var optHist = {
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis' }),
      legend: { data: ['训练期', '验证期'], textStyle: { color: muted }, top: 8 },
      grid: { left: 56, right: 20, top: 46, bottom: 34 },
      xAxis: Object.assign(axisBase(true), { type: 'category', data: midPoints, name: '标签值', nameTextStyle: { color: muted }, axisLabel: { color: muted, fontSize: 10, interval: 4 } }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: '样本数', nameTextStyle: { color: muted } }),
      series: [
        { name: '训练期', type: 'bar', data: D.label.histogram.train.counts, itemStyle: { color: accent, opacity: 0.85 }, barGap: '10%' },
        { name: '验证期', type: 'bar', data: D.label.histogram.valid.counts, itemStyle: { color: accent2, opacity: 0.75 } }
      ]
    };
    chartHist.setOption(optHist);
  }

  // ---------- 图 3-2 / 3-3 标签截面均值与标准差 ----------
  function renderLabelTime(id, seriesName, dataKey, color, yMin, yMax) {
    var chart = init(id);
    if (!chart) { return; }
    var ts = D.label.time_series;
    var opt = {
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis' }),
      grid: { left: 56, right: 20, top: 30, bottom: 42 },
      xAxis: Object.assign(axisBase(true), { type: 'category', boundaryGap: false, data: ts.time }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: seriesName, nameTextStyle: { color: muted }, min: yMin, max: yMax }),
      series: [{ name: seriesName, type: 'line', data: ts[dataKey], smooth: true, showSymbol: false, lineStyle: { width: 1.6, color: color }, itemStyle: { color: color }, areaStyle: { color: color, opacity: 0.06 } }]
    };
    opt.xAxis.markLine = {
      silent: true, symbol: 'none',
      lineStyle: { color: ink, type: 'solid', width: 1, opacity: 0.45 },
      data: [{ xAxis: 486 }, { xAxis: 2918 }, { xAxis: 3161 }]
    };
    addSplitBandsPct(opt, ts.time);
    chart.setOption(opt);
  }
  renderLabelTime('chart-label-mean', '截面均值', 'mean', accent, 0.45, 0.55);
  renderLabelTime('chart-label-std', '截面标准差', 'std', accent2, 0.26, 0.32);

  // ---------- 水平条形 Top-N 工具 ----------
  function renderHBar(id, title, dataList, colorFn, xName, formatter) {
    var chart = init(id);
    if (!chart || !dataList.length) { return; }
    var items = dataList.slice().reverse();
    var opt = {
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: formatter }),
      grid: { left: 74, right: 46, top: 12, bottom: 30 },
      xAxis: Object.assign(axisBase(false), { type: 'value', name: xName, nameTextStyle: { color: muted } }),
      yAxis: Object.assign(axisBase(true), { type: 'category', data: items.map(function (d) { return d.feature; }) }),
      series: [{
        type: 'bar',
        data: items.map(function (d) { return d.value; }),
        barWidth: 12,
        itemStyle: { color: function (p) { return colorFn ? colorFn(p) : accent; }, borderRadius: [0, 3, 3, 0] },
        label: { show: true, position: 'right', color: muted, fontSize: 10, formatter: function (p) { return formatter ? formatter(p.value) : p.value; } }
      }]
    };
    chart.setOption(opt);
  }

  // ---------- 图 4-1 max_abs ----------
  renderHBar('chart-max-abs', '', D.numeric.top_max_abs, null, '最大绝对值', function (v) { return v.toFixed(1); });

  // ---------- 图 4-2 训练期 IQR 异常率 ----------
  renderHBar('chart-outlier', '', D.numeric.top_outlier_train, function () { return accent2; }, 'IQR 异常率', function (v) { return (v * 100).toFixed(1) + '%'; });

  // ---------- 图 4-3 验证期均值漂移（正负分色） ----------
  renderHBar('chart-shift', '', D.numeric.top_shift_valid, function (p) { return p.value >= 0 ? accent : '#c94f3d'; }, '均值漂移（训练标准差）', function (v) { return v.toFixed(2); });

  // ---------- 图 4-4 测试期 PSI ----------
  renderHBar('chart-psi', '', D.numeric.top_psi_test, function () { return '#8a6d1a'; }, 'PSI', function (v) { return v.toFixed(2); });

  // ---------- 图 4-5 漂移分布直方图 ----------
  var chartDrift = init('chart-drift-dist');
  if (chartDrift) {
    function histData(obj) {
      var out = [];
      for (var i = 0; i < obj.edges.length - 1; i++) {
        out.push([obj.edges[i], obj.edges[i + 1], obj.counts[i]]);
      }
      return out;
    }
    var dd = D.numeric.drift_dist;
    var optDrift = {
      animation: false,
      tooltip: { trigger: 'axis', appendToBody: true, backgroundColor: bg2, borderColor: rule, textStyle: { color: ink } },
      legend: { data: ['最大 PSI', '最大 |均值漂移|'], textStyle: { color: muted }, top: 8 },
      grid: { left: 56, right: 20, top: 46, bottom: 34 },
      xAxis: Object.assign(axisBase(true), { type: 'value', name: '漂移指标值', nameTextStyle: { color: muted } }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: '特征数', nameTextStyle: { color: muted } }),
      series: [
        { name: '最大 PSI', type: 'bar', data: histData(dd.max_psi), itemStyle: { color: accent, opacity: 0.75 } },
        { name: '最大 |均值漂移|', type: 'bar', data: histData(dd.max_shift), itemStyle: { color: accent2, opacity: 0.65 } }
      ]
    };
    chartDrift.setOption(optDrift);
  }

  // ---------- 图 5-1 RankIC Top15 双系列 ----------
  var chartRankic = init('chart-rankic-top');
  if (chartRankic) {
    var rankTop = D.rankic.top_valid.slice().reverse();
    var optRankic = {
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: function (v) { return v.toFixed(4); } }),
      legend: { data: ['训练期抽样', '验证期'], textStyle: { color: muted }, top: 8 },
      grid: { left: 74, right: 46, top: 42, bottom: 30 },
      xAxis: Object.assign(axisBase(false), { type: 'value', name: '平均 RankIC', nameTextStyle: { color: muted }, axisLabel: { color: muted, formatter: function (v) { return v.toFixed(2); } } }),
      yAxis: Object.assign(axisBase(true), { type: 'category', data: rankTop.map(function (d) { return d.feature; }) }),
      series: [
        { name: '训练期抽样', type: 'bar', data: rankTop.map(function (d) { return d.train_sample; }), barWidth: 8, itemStyle: { color: muted, borderRadius: [0, 3, 3, 0] } },
        { name: '验证期', type: 'bar', data: rankTop.map(function (d) { return d.valid; }), barWidth: 8, itemStyle: { color: function (p) { return p.value >= 0 ? accent : '#c94f3d'; }, borderRadius: [0, 3, 3, 0] } }
      ]
    };
    chartRankic.setOption(optRankic);
  }

  // ---------- 图 5-2 |RankIC| 分布直方图 ----------
  var chartRankicDist = init('chart-rankic-dist');
  if (chartRankicDist) {
    var rd = D.rankic.dist;
    var rdData = [];
    for (var rdi = 0; rdi < rd.edges.length - 1; rdi++) { rdData.push([rd.edges[rdi], rd.edges[rdi + 1], rd.counts[rdi]]); }
    chartRankicDist.setOption({
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis' }),
      grid: { left: 56, right: 20, top: 24, bottom: 34 },
      xAxis: Object.assign(axisBase(true), { type: 'value', name: '|平均 RankIC|', nameTextStyle: { color: muted } }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: '特征数', nameTextStyle: { color: muted } }),
      series: [{ type: 'bar', data: rdData, itemStyle: { color: accent, opacity: 0.8 } }]
    });
  }

  // ---------- 图 6-1 类别基数（对数） ----------
  var chartCatCard = init('chart-cat-cardinality');
  if (chartCatCard) {
    var catCard = D.category.cardinality;
    chartCatCard.setOption({
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis', axisPointer: { type: 'shadow' } }),
      grid: { left: 60, right: 46, top: 12, bottom: 30 },
      xAxis: Object.assign(axisBase(false), { type: 'value', name: '基数（对数）', nameTextStyle: { color: muted }, axisLabel: { color: muted, formatter: function (v) { return v >= 1000 ? (v / 1000).toFixed(1) + 'k' : v; } } }),
      yAxis: Object.assign(axisBase(true), { type: 'category', data: catCard.map(function (d) { return d.feature; }) }),
      series: [{
        type: 'bar',
        data: catCard.map(function (d) { return d.value; }),
        barWidth: 14,
        itemStyle: { color: function (p) { return p.value >= 1000 ? '#c94f3d' : accent; }, borderRadius: [0, 3, 3, 0] },
        label: { show: true, position: 'right', color: muted, fontSize: 10, formatter: function (p) { return p.value >= 1000 ? p.value.toLocaleString() : p.value; } }
      }]
    });
  }

  // ---------- 图 6-2 未见类别率 ----------
  var chartCatUnseen = init('chart-cat-unseen');
  if (chartCatUnseen) {
    var catUnseen = D.category.unseen;
    chartCatUnseen.setOption({
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: function (v) { return (v * 100).toFixed(2) + '%'; } }),
      legend: { data: ['验证期', '测试期'], textStyle: { color: muted }, top: 8 },
      grid: { left: 60, right: 46, top: 42, bottom: 30 },
      xAxis: Object.assign(axisBase(false), { type: 'value', name: '未见类别率', nameTextStyle: { color: muted }, axisLabel: { color: muted, formatter: function (v) { return (v * 100).toFixed(0) + '%'; } } }),
      yAxis: Object.assign(axisBase(true), { type: 'category', data: catUnseen.map(function (d) { return d.feature; }) }),
      series: [
        { name: '验证期', type: 'bar', data: catUnseen.map(function (d) { return d.valid; }), barWidth: 8, itemStyle: { color: accent2, borderRadius: [0, 3, 3, 0] } },
        { name: '测试期', type: 'bar', data: catUnseen.map(function (d) { return d.test; }), barWidth: 8, itemStyle: { color: '#c94f3d', borderRadius: [0, 3, 3, 0] } }
      ]
    });
  }

  // ---------- 图 6-3 类别变化率分组柱状 ----------
  var chartCatChange = init('chart-cat-change');
  if (chartCatChange) {
    var catChange = D.category.change_list;
    var changeStages = ['pretrain', 'train', 'valid', 'test'];
    var changeLabels = { pretrain: '预训练期', train: '训练期', valid: '验证期', test: '测试期' };
    chartCatChange.setOption({
      animation: false,
      tooltip: Object.assign(commonTooltip(), { trigger: 'axis', valueFormatter: function (v) { return (v * 100).toFixed(1) + '%'; } }),
      legend: { data: changeStages.map(function (s) { return changeLabels[s]; }), textStyle: { color: muted }, top: 8 },
      grid: { left: 56, right: 20, top: 46, bottom: 30 },
      xAxis: Object.assign(axisBase(true), { type: 'category', data: catChange.map(function (d) { return d.feature; }) }),
      yAxis: Object.assign(axisBase(false), { type: 'value', name: '变化率', nameTextStyle: { color: muted }, axisLabel: { color: muted, formatter: function (v) { return (v * 100).toFixed(0) + '%'; } } }),
      series: [
        { name: '预训练期', type: 'bar', data: catChange.map(function (d) { return d.pretrain; }), itemStyle: { color: muted, opacity: 0.7 } },
        { name: '训练期', type: 'bar', data: catChange.map(function (d) { return d.train; }), itemStyle: { color: accent } },
        { name: '验证期', type: 'bar', data: catChange.map(function (d) { return d.valid; }), itemStyle: { color: accent2 } },
        { name: '测试期', type: 'bar', data: catChange.map(function (d) { return d.test; }), itemStyle: { color: '#c9a227' } }
      ]
    });
  }
})();
