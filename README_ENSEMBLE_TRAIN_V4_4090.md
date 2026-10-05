# Semifinal Ensemble V4 YOLO11m Training

This package trains a second visual detector for a two-model ensemble. It does
not modify the existing YOLO11s model or its current 66.16 submission.

## Design

- Model B: public YOLO11m pretrained checkpoint bundled in this package
- Input size: 1280
- Phase 1: 45 epochs, AdamW, moderate augmentation
- Phase 2: 8 epochs, low learning rate, no mosaic
- DataLoader workers: 0 to avoid the known shared-memory bus error
- Checkpoint evaluation: frozen development split at 1280
- Selection score: `20*precision + 60*recall + 20*mAP50`
- Output: one selected Model B checkpoint and compact result archive

## 1. Extract

```bash
cd /mnt/proj
python3 -m zipfile -e \
  semifinal_ensemble_train_v4_20260926.zip \
  /mnt/proj
cd /mnt/proj/semifinal_ensemble_train_v4_20260926
```

## 2. Preflight

```bash
/mnt/proj/iron/.venv/bin/python \
  scripts/run_ensemble_v4_training_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

The preflight must report an RTX 4090, at least 18 GiB free VRAM, the expected
YOLO11m SHA256, and `dataloader_workers: 0`.

## 3. Start or resume the complete pipeline

```bash
cd /mnt/proj/semifinal_ensemble_train_v4_20260926

CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_ensemble_v4_training_pipeline.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee ensemble_v4_training.log
```

Run the same command again after an interruption. Completed stages are reused,
and an incomplete training stage resumes from its `last.pt` checkpoint.

## 4. Final outputs

```text
/mnt/proj/iron/runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt
/mnt/proj/iron/runs/semifinal/ensemble_v4_yolo11m_pipeline/final/FINAL.json
/mnt/proj/semifinal_ensemble_v4_yolo11m_results_20260926.tar.gz
```

Download the result archive after the pipeline reports `status: complete`.

## 5. Monitor

```bash
nvidia-smi
```

```bash
tail -f \
  /mnt/proj/iron/runs/semifinal/ensemble_v4_yolo11m_pipeline/phase1_train.log
```

Do not start the TTA inference pipeline while training is active.
