# Ensemble V6: consensus now, native TTA development gate next

This package never trains a model. It has two independent stages:

1. Reuse the existing Model A/B Test caches to build the source-aware consensus
   submission selected on both frozen development views.
2. Run native multi-scale and flip TTA for Models A and B on the complete frozen
   development split. Return the result archive before any TTA Test inference.

The protected official result is 66.96. Do not replace it with a lower score.

## Extract

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_ensemble_v6_gpu_20260927.zip /mnt/proj
cd /mnt/proj/semifinal_ensemble_v6_gpu_20260927
```

## Stage 1: source-aware consensus, no GPU inference

```bash
/mnt/proj/iron/.venv/bin/python \
  scripts/run_ensemble_v6_consensus_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  --preflight-only

/mnt/proj/iron/.venv/bin/python -u \
  scripts/run_ensemble_v6_consensus_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  2>&1 | tee ensemble_v6_consensus.log
```

Output:

```text
/mnt/proj/iron/runs/semifinal/ensemble_v6_consensus_submission/semifinal_ensemble_v6_consensus.zip
```

The frozen-development proxy improved from 68.297 to 68.367 while preserving
all matched true positives on both views. This is a small, not guaranteed gain.

## Stage 2: full native TTA development evaluation

Preflight:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python \
  scripts/run_ensemble_v6_tta_dev_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

Run:

```bash
cd /mnt/proj/semifinal_ensemble_v6_gpu_20260927
set -o pipefail

CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_ensemble_v6_tta_dev_pipeline.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee ensemble_v6_tta_dev.log
```

This evaluates four complete cells: Model A/B times original/grid-crop view.
Completed cells are reused after interruption; only the active incomplete cell
restarts. Model A uses 1024 and Model B uses 1280, batch 1, FP16, native TTA.

Return:

```text
/mnt/proj/ensemble_v6_tta_dev_results_20260927.tar.gz
```

Do not launch TTA Test inference yet. The returned development caches are used
to select among A-TTA+B, A+B-TTA, dual-TTA, and plain/TTA multi-source fusion.

