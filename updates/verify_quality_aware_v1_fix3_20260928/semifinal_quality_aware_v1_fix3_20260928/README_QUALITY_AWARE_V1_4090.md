# Quality-aware detector V1: matched-compute RTX 4090 experiment

This package tests one detector-level hypothesis after SAFE-R2 post-processing
failed its release gate. It does not use Test or the final lockbox and it never
creates a submission.

## Hypothesis

Ultralytics 8.3.169 already produces task-aligned soft classification targets
whose normalized alignment score depends on both class confidence and overlap;
they are not a pure IoU target. Stock YOLO11 optimizes them with plain BCE. The
experimental arm applies an elementwise quality-weighted Varifocal loss:

```text
positive weight = matched quality
negative weight = 0.75 * sigmoid(logit)^2
```

The output head is unchanged. The custom criterion is injected only into the
live training model after the stock EMA is created. Resulting checkpoints are
ordinary Ultralytics `DetectionModel` weights; a fresh-process CPU reload and
finite forward pass is mandatory before evaluation.

## Controlled experiment

Both arms start from the protected Model A checkpoint with SHA256:

```text
d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d
```

They use the same balanced-v3 training data, seed, optimizer, LR schedule,
augmentation, 1024 resolution, batch 8, and 8 epochs:

- `bce_control`: ordinary BCE continuation;
- `varifocal`: the quality-aware loss.

Only the fixed epoch-8 terminal checkpoint and untouched protected starting
point are evaluated on the frozen 459-source development split in original and
grid-crop views. Both arms use the preregistered export threshold `0.0001`;
there is no checkpoint or threshold scan. Official duplicate groups remain
frozen by the existing split manifest.

Stage 1 passes only if Varifocal beats the matched BCE control by at least
`+0.20` weighted score, at least `+0.10` score, `+0.005` mAP50 and `+0.005`
AP75 in each view, with no mAP50:95 regression, no per-class AP50 drop beyond
`0.02`, no recall drop beyond `0.002`, and no FP/image increase beyond 5%.
Optimizer updates, microbatches, epochs, dataset signatures, code signatures,
and exact protected-weight transfer must also match. A pass authorizes only an
exact V7 TTA fusion evaluation with frozen Model B; it does not authorize Test.

## Extract

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_quality_aware_v1_fix3_20260928.zip /mnt/proj
cd /mnt/proj/semifinal_quality_aware_v1_fix3_20260928
```

## 1. Preflight

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_quality_aware_experiment.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

Preflight requires RTX 4090, at least 10 GiB free VRAM, Ultralytics `8.3.169`,
the protected Model A hash, the official split hash, and the package manifest.

## 2. One-epoch smoke run

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_quality_aware_experiment.py \
  --project-root /mnt/proj/iron \
  --smoke-only
```

The smoke run uses 2% of training data and writes to an isolated directory. It
checks CUDA forward/backward, the custom loss, optimizer updates, checkpoint
serialization, and the standard YOLO11 model layout.

## 3. Full matched-compute experiment

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_quality_aware_experiment.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee quality_aware_v1.log
```

The runner is resumable from each arm's `last.pt`, refuses incompatible resume
contracts, evaluates the protected baseline in both arms, and records optimizer
step/assignment-quality witnesses. Stale training or evaluation caches fail on
code/data/checkpoint signatures. The result archive includes copied metrics,
fresh-process compatibility reports, training witnesses, an offline audit
index, filtered predictions, compact ground truth, the frozen split manifest,
and a per-file SHA256 manifest. Stage-1 AP/FP/per-class metrics can therefore
be recomputed without the cloud run directory. It will not overwrite an
existing final result archive.

Return:

```text
/mnt/proj/quality_aware_v1_fix3_results_20260928.tar.gz
```

If `QUALITY_AWARE_DECISION.json` says `STOP`, do not submit either checkpoint.
If it says `PASS_STAGE1`, the next required step is exact V7 native-TTA fusion
evaluation with frozen Model B
`02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090`.
That Stage 2 must reproduce the exact V7 baseline, add paired group bootstrap
and score-quality calibration checks, and pass the separate final-lockbox gate
before any Test action.

