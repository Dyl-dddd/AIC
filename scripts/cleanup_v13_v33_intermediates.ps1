param([switch]$Apply)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path.TrimEnd('\')
$archive = 'D:\新建文件夹 (8)\semifinal_v13_hires_experts_results_20261001.tar.gz'
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw "V13 original archive is missing; refusing to remove its extracted copy: $archive"
}

$directories = @(
    'analysis\v13_hires_experts_results_20261001',
    'runs\semifinal\v20_crop_verifier_20261002\crops',
    'runs\semifinal\v26_zonglie_context_localcheck\real_smoke'
)

foreach ($relative in $directories) {
    $path = Join-Path $projectRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Container)) { continue }
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes project root: $resolved"
    }
    $links = @(Get-ChildItem -LiteralPath $resolved -Recurse -Force -Attributes ReparsePoint -ErrorAction SilentlyContinue)
    if ($links.Count -gt 0) { throw "Nested link/reparse point under $resolved" }
    $files = @(Get-ChildItem -LiteralPath $resolved -Recurse -File -Force)
    $bytes = ($files | Measure-Object Length -Sum).Sum
    Write-Output ("{0}: {1:N2} GiB, {2} files" -f $resolved, ($bytes / 1GB), $files.Count)
    if ($Apply) { Remove-Item -LiteralPath $resolved -Recurse -Force }
}

$weights = Join-Path $projectRoot 'runs\local4060\stock4060_60ep_b1_noval_20260902\weights'
$resolvedWeights = (Resolve-Path -LiteralPath $weights).Path
if (-not $resolvedWeights.StartsWith($projectRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw "Weights directory escapes project root: $resolvedWeights"
}
if (-not (Test-Path -LiteralPath (Join-Path $resolvedWeights 'best.pt') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $resolvedWeights 'last.pt') -PathType Leaf)) {
    throw 'best.pt or last.pt is missing; refusing to delete old epoch checkpoints'
}
$epochs = @(Get-ChildItem -LiteralPath $resolvedWeights -File | Where-Object { $_.Name -match '^epoch\d+\.pt$' })
$epochBytes = ($epochs | Measure-Object Length -Sum).Sum
Write-Output ("{0}: {1:N2} GiB, {2} intermediate checkpoints" -f $resolvedWeights, ($epochBytes / 1GB), $epochs.Count)
if ($Apply) {
    foreach ($file in $epochs) {
        if (-not $file.FullName.StartsWith($resolvedWeights + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unexpected checkpoint path: $($file.FullName)"
        }
        Remove-Item -LiteralPath $file.FullName -Force
    }
    Write-Output 'Cleanup complete. Original data, V12 submission, best.pt, and last.pt were not targeted.'
} else {
    Write-Output 'Dry run only. Re-run with -Apply to permanently remove the listed files.'
}
