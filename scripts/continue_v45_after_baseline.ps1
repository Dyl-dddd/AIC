param([int]$BaselinePid = 9756)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$python = 'D:\Aca\python.exe'
$probe = Join-Path $root 'scripts\probe_v45_contrast_preprocessing.py'
$baselineResult = Join-Path $root 'fresh_20261004\local_baseline_result.json'
$output = Join-Path $root 'runs\semifinal\v45_contrast_preprocessing_20261005'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$status = Join-Path $output 'launch_status.log'

function Log($message) {
    Add-Content -LiteralPath $status -Value ("{0} {1}" -f (Get-Date -Format o), $message)
}

Log "Waiting for existing baseline PID $BaselinePid"
while (Get-Process -Id $BaselinePid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 15
}
if (-not (Test-Path -LiteralPath $baselineResult -PathType Leaf)) {
    Log 'Baseline process exited without its result manifest; refusing to run V45.'
    exit 1
}
Log 'Baseline result manifest exists. Checking local GPU and disk.'
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    $freeText = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
    $freeMiB = [int]($freeText | Select-Object -First 1).Trim()
    $diskFree = (Get-PSDrive D).Free
    if ($freeMiB -ge 5500 -and $diskFree -ge 1GB) { break }
    Start-Sleep -Seconds 15
}
if ($freeMiB -lt 5500 -or $diskFree -lt 1GB) {
    Log "Resource gate failed: GPU free $freeMiB MiB, D free $diskFree bytes."
    exit 2
}

Set-Location -LiteralPath $root
Log "Launching four-image smoke; GPU free $freeMiB MiB."
& $python -u $probe --limit 4 *> (Join-Path $output 'smoke.log')
if ($LASTEXITCODE -ne 0) {
    Log "Smoke failed with exit code $LASTEXITCODE; full run not started."
    exit $LASTEXITCODE
}
Log 'Smoke passed. Launching all 1757 numeric development tiles.'
& $python -u $probe *> (Join-Path $output 'full.log')
if ($LASTEXITCODE -ne 0) {
    Log "Full probe failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
Log 'Full probe complete; see report.json. No Test images or labels were used.'
