# SAFE-R2 one-time final-lockbox evaluation

This package evaluates one already-frozen SAFE-R2 profile against the exact V7
baseline on the untouched official `final` holdout. It performs no training,
parameter search, threshold sweep, or leaderboard submission.

The included profile is intentionally fail-closed. The runner refuses to read
the final holdout unless `configs/inference/safe_r2_frozen_profile.json` was
replaced **before packaging** by an immutable group-OOF profile containing all
of these metadata values:

```json
{
  "method": "SAFE-R2",
  "deployment_allowed": true,
  "gate": "PASS",
  "training_view": "original",
  "test_labels_used": false
}
```

The currently selected SAFE-R2 experiment did not pass that gate, so the
provided package is a prepared protocol, not authorization to open the
lockbox. Do not change `deployment_allowed` by hand.

## Fixed protocol

- Official split SHA256:
  `c46359024fa2c06b4d03cef91e274588cf555a0d053108b840010df3a6573bf9`
- 291 final-holdout source images and 517 non-`qilie` ground-truth boxes.
- Protected Model A/B hashes are verified.
- Both models run native TTA at 1024/1280 on original and 2x3 grid views.
- V7 is replayed with fixed IoU 0.70, boost 1.10, anchor coordinates,
  unmatched-B factor 0.50, B scale 0.35, and final NMS 0.72.
- SAFE-R2 changes no prediction membership; it only reranks and, if the frozen
  profile enables it, applies guarded refinement.
- The same profile is applied to both views with no search.

The pre-registered lockbox gate requires weighted score delta at least +0.15,
no score or mAP50 decrease in either view, recall drop no worse than 0.002 in
either view, and exactly unchanged prediction counts. Any failure is `STOP`.

## Extract and preflight

```bash
cd /mnt/proj
python3 -m zipfile -e safe_r2_final_lockbox_20260928.zip /mnt/proj
cd /mnt/proj/safe_r2_final_lockbox_20260928

CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python \
  scripts/run_safe_r2_final_lockbox_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

With the included STOP profile, preflight must fail closed before inference.
Only a genuinely passed, frozen OOF profile may replace it, and the release ZIP
must then be rebuilt so its manifest binds the exact profile bytes.

## One-time run

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 /mnt/proj/iron/.venv/bin/python -u \
  scripts/run_safe_r2_final_lockbox_pipeline.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee safe_r2_final_lockbox.log
```

Return only this archive for independent review:

```text
/mnt/proj/safe_r2_final_lockbox_results_20260928.tar.gz
```

`LOCKBOX_DECISION.json` is terminal for the frozen profile. If it says STOP,
do not inspect error cases, edit the profile, or rerun against final labels.
This package never creates a submission JSON or submission ZIP.
