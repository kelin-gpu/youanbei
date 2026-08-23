# 一键启动 exp_021（自动设置训练授权，等价于手动设置两个环境变量）
# 用法:  & .\start_exp021.ps1 [fit|submit|all]    （默认 fit）

$stage = if ($args.Count -gt 0) { $args[0] } else { "fit" }

$env:DSCR_EXP016_MODE = "full"
$env:DSCR_EXP016_ALLOW_TRAINING = "YES"

Write-Host "[start_exp021] mode=$env:DSCR_EXP016_MODE allow_training=$env:DSCR_EXP016_ALLOW_TRAINING stage=$stage"

& D:\anaconda\anaconda_data\envs\jingge_ts\python.exe d:/google_dl/book/youanbei/02_experiments/exp_021_retrain_head_router/run_exp021.py $stage
exit $LASTEXITCODE
