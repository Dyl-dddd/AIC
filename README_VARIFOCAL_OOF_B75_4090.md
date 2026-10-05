# Varifocal OOF B75 result-level rescoring

This package keeps the V7 primary prediction membership and coordinates exactly
unchanged.  It runs the returned Varifocal checkpoint once on Test, matches its
quality evidence to V7 boxes at IoU 0.50, and changes only ranking scores using a
five-fold OOF-tested calibration rule.

Development OOF proxy:

- V7 weighted: `68.859562`
- rescored weighted: `69.004355`
- delta: `+0.144793`
- original delta: `+0.130035`
- grid-crops delta: `+0.168799`
- TP, FP, box membership and coordinates: unchanged

The preregistered `+0.20` release gate was **not** met.  Treat the output as an
experimental spare-slot submission, not a guaranteed replacement for the
official V7 score of 67.49.

## Run on the same RTX 4090 cloud project

Extract the ZIP, enter its top-level directory, then run:

```bash
python -u scripts/run_varifocal_oof_rescore_pipeline.py \
  --project-root /mnt/proj/iron
```

The pipeline:

1. verifies the Varifocal checkpoint SHA256;
2. reuses or builds the verified V7 primary submission;
3. runs Varifocal with the exact development inference policy (no TTA);
4. rescales V7 scores without adding, removing, or moving any box;
5. validates all 788 image IDs and writes a one-file submission ZIP.

Expected output:

`/mnt/proj/iron/runs/semifinal/varifocal_oof_b75_submission/semifinal_varifocal_oof_b75.zip`

Keep `semifinal_ensemble_v7_tta_primary.zip` as the protected best.  If a spare
leaderboard attempt is available, submit the new ZIP once and retain it only if
the official score exceeds 67.49.
