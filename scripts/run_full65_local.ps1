param(
    [string]$Python = "D:\Aca\python.exe",
    [string]$BaselineName = "baseline4060_full65_b1_noval_20260902",
    [string]$LongName = "stock4060_60ep_b1_noval_20260902"
)

$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
$logRoot = Join-Path $workspace "runs\local4060\_launcher"
$launcherLog = Join-Path $logRoot "full65_b1_noval_20260902.log"
$selection = Join-Path $workspace "runs\local4060\$BaselineName\original_selection.json"

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Set-Location -LiteralPath $workspace
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:CUDA_VISIBLE_DEVICES = "0"

"[$(Get-Date -Format o)] Starting 5-epoch baseline: $BaselineName" | Tee-Object -FilePath $launcherLog -Append
& $Python -u scripts\local_cloud.py run `
    --stage baseline `
    --name $BaselineName `
    --workers 0 2>&1 | Tee-Object -FilePath $launcherLog -Append
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format o)] Baseline failed with exit code $LASTEXITCODE; long run was not started." | Tee-Object -FilePath $launcherLog -Append
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $selection)) {
    "[$(Get-Date -Format o)] Baseline selection is missing: $selection" | Tee-Object -FilePath $launcherLog -Append
    exit 2
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
