param(
    [string]$Python = "D:\Aca\python.exe",
    [string]$BaselineName = "baseline4060_full65_b1_noval_20260902",
    [string]$LongName = "stock4060_60ep_b1_noval_20260902"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspace "runs\local4060\_launcher"
$launcherLog = Join-Path $logRoot "resume_full65_b1_noval_20260902.log"
$selection = Join-Path $workspace "runs\local4060\$BaselineName\original_selection.json"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Set-Location -LiteralPath $workspace
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:CUDA_VISIBLE_DEVICES = "0"

"[$(Get-Date -Format o)] Recovering baseline evaluations from completed checkpoints: $BaselineName" | Tee-Object -FilePath $launcherLog -Append
& $Python -u scripts\local_cloud.py evaluate-run `
    --name $BaselineName 2>&1 | Tee-Object -FilePath $launcherLog -Append
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format o)] Baseline evaluation recovery failed with exit code $LASTEXITCODE." | Tee-Object -FilePath $launcherLog -Append
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $selection)) {
    "[$(Get-Date -Format o)] Baseline selection is missing: $selection" | Tee-Object -FilePath $launcherLog -Append
    exit 2
}
if (Test-Path -LiteralPath (Join-Path $workspace "runs\local4060\$LongName")) {
    "[$(Get-Date -Format o)] Long-run target already exists; refusing overwrite: $LongName" | Tee-Object -FilePath $launcherLog -Append
    exit 3
}

"[$(Get-Date -Format o)] Starting fresh 60-epoch long run: $LongName" | Tee-Object -FilePath $launcherLog -Append
& $Python -u scripts\local_cloud.py run `
    --stage long `
    --name $LongName `
    --epochs 60 `
    --after $selection `
    --workers 0 2>&1 | Tee-Object -FilePath $launcherLog -Append
$exitCode = $LASTEXITCODE

if ($exitCode -eq 0) {
    "[$(Get-Date -Format o)] Full 65-epoch workflow completed." | Tee-Object -FilePath $launcherLog -Append
} else {
    "[$(Get-Date -Format o)] Long run failed with exit code $exitCode." | Tee-Object -FilePath $launcherLog -Append
}
exit $exitCode
