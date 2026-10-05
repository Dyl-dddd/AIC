# Ensemble V7: dual-model native TTA Test inference

This package performs no training. It runs the two protected checkpoints with
native multi-scale/flip TTA on all 788 Test images, fuses their raw candidates,
validates the result, and creates two submission ZIP files.

The frozen dual-view development proxy improved from V6 `68.367` to `68.860`.
This is not an official-score guarantee. Keep the protected official score
`66.96` unless the new result is higher.

## Extract

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_ensemble_v7_tta_20260927.zip /mnt/proj
cd /mnt/proj/semifinal_ensemble_v7_tta_20260927
```

## Preflight

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python \
  scripts/run_ensemble_v7_tta_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  --preflight-only
```

The preflight must report RTX 4090, 788 uniquely named Test images, the exact
Model A/B SHA256 values, `tta: true`, and `training: false`.

## Run

```bash
cd /mnt/proj/semifinal_ensemble_v7_tta_20260927
set -o pipefail

CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_ensemble_v7_tta_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  2>&1 | tee ensemble_v7_tta.log
```

The pipeline reuses a verified Model A TTA cache when available. Completed
caches are reusable after interruption; an incomplete temporary cache is left
untouched for diagnosis and is never accepted as complete.

## Outputs and submission order

Submit this first:

```text
/mnt/proj/iron/runs/semifinal/ensemble_v7_tta_submission/semifinal_ensemble_v7_tta_primary.zip
```

The primary configuration had the highest robust development proxy:

|System|Original|Grid crops|Weighted|Delta vs V6|
|---|---:|---:|---:|---:|
|V6 non-TTA consensus|68.418|68.283|68.367|--|
|V7 TTA primary|68.703|69.114|68.860|+0.493|

The conservative fallback emits fewer predictions on development data and is:

```text
/mnt/proj/iron/runs/semifinal/ensemble_v7_tta_submission/semifinal_ensemble_v7_tta_conservative.zip
```

Do not submit the conservative fallback automatically. Use it only after the
primary official result is known and only if submission quota remains.

Return this summary together with the primary official score:

```text
/mnt/proj/iron/runs/semifinal/ensemble_v7_tta_submission/ENSEMBLE_V7_TTA_SUBMISSION_SUMMARY.json
```
