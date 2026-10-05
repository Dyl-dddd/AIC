# Semifinal TTA Inference Package

This package runs inference only. It does not train or modify the model.

TTA is enabled through the Ultralytics detection model with `augment=True`.
The model evaluates multiple scaled and flipped views, maps detections back to
the original tile coordinates, and then applies the existing cross-tile NMS.

The non-TTA candidate cache is never overwritten. TTA writes to:

```text
/mnt/proj/iron/runs/semifinal/final_submission_tta_v1/
```

## 1. Extract

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_tta_inference_v1_20260925.zip /mnt/proj
cd /mnt/proj/semifinal_tta_inference_v1_20260925
```

## 2. Preflight

```bash
/mnt/proj/iron/.venv/bin/python scripts/run_tta_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  --preflight-only
```

The preflight must report 788 images, the expected weight SHA256, an RTX 4090,
and `"tta": true`.

## 3. Run TTA inference

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python \
  scripts/run_tta_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --source /mnt/proj/iron/data/semifinal/test \
  2>&1 | tee tta_inference.log
```

TTA is slower than the previous inference because each image is evaluated with
multiple transformed views. Batch size is fixed at 1 to keep GPU memory usage
bounded on the RTX 4090.

## 4. Outputs

```text
/mnt/proj/iron/runs/semifinal/final_submission_tta_v1/semifinal_tta_recall_safe.zip
/mnt/proj/iron/runs/semifinal/final_submission_tta_v1/semifinal_tta_dev_calibrated.zip
/mnt/proj/iron/runs/semifinal/final_submission_tta_v1/TTA_SUBMISSION_SUMMARY.json
```

Submit `semifinal_tta_recall_safe.zip` first for a direct comparison with the
current 65.65 non-TTA result.

If an interrupted temporary cache exists, preserve it before restarting:

```bash
mv \
  /mnt/proj/iron/runs/semifinal/final_submission_tta_v1/test_candidates_tta_1e5.tmp.jsonl.gz \
  /mnt/proj/iron/runs/semifinal/final_submission_tta_v1/test_candidates_tta_1e5.interrupted.jsonl.gz
```
