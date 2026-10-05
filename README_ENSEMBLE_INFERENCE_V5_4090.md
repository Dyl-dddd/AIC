# Ensemble V5 4090 inference

This package runs no training. It combines the protected YOLO11s Model A and the new
YOLO11m Model B after both checkpoints passed hash and frozen-development checks.

The bounded development result improved from 66.640 to 68.297 with the simple global
fusion. The class-routed result reached 68.409 on the same development set, but it is a
more optimistic diagnostic and must not be submitted before the global candidate is
tested. These values are not official Test scores.

Before any official submission, confirm that the competition permits multi-model
inference. The public task page does not explicitly ban it, but any separate rule PDF or
organizer clarification takes precedence.

## Extract

```bash
cd /mnt/proj

python3 -m zipfile -e \
  semifinal_ensemble_inference_v5_20260927.zip \
  /mnt/proj

cd /mnt/proj/semifinal_ensemble_inference_v5_20260927
```

## Preflight

```bash
/mnt/proj/iron/.venv/bin/python \
  scripts/run_ensemble_v5_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  --preflight-only
```

Expected checks include 788 images, RTX 4090, Model A SHA beginning with `d69b236a`, and
Model B SHA beginning with `02f1f09b`.

## Run

```bash
cd /mnt/proj/semifinal_ensemble_inference_v5_20260927
set -o pipefail

CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_ensemble_v5_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  2>&1 | tee ensemble_v5.log
```

The existing non-TTA Model A cache is reused when valid. Model B runs once at 1280.
Rerunning the same command reuses verified caches and submissions.

## Outputs

```bash
cat \
  /mnt/proj/iron/runs/semifinal/ensemble_v5_submission/ENSEMBLE_V5_SUBMISSION_SUMMARY.json

ls -lh \
  /mnt/proj/iron/runs/semifinal/ensemble_v5_submission/*.zip
```

Candidate order, only if multi-model inference is permitted:

1. `semifinal_ensemble_v5_global_fusion.zip`
2. `semifinal_ensemble_v5_class_routed.zip`

Do not replace the protected 66.16 result unless the official score is higher.
