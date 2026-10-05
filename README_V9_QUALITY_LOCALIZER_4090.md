# V9 YOLO11m quality-localizer experiment

This is a development-only experiment package.  It never reads Test and never
creates a leaderboard submission.

It starts two matched 10-epoch refinements from the exact protected YOLO11m
Model B checkpoint.  Both arms strengthen box and DFL supervision; the only
experimental difference is the Varifocal quality target power (`1.0` vs `2.0`).
The current Model B endpoint is evaluated in the same fixed-threshold run.

Run on the existing RTX 4090 project:

```bash
cd /mnt/proj/iron
unzip -q semifinal_v9_quality_localizer_20260930.zip
cd semifinal_v9_quality_localizer_20260930
python -u scripts/run_v9_quality_localizer_experiment.py \
  --project-root /mnt/proj/iron
```

The run is resumable.  Expected result archive:

```text
/mnt/proj/iron/runs/semifinal_v9_quality_localizer_results_20260930.tar.gz
```

Download that archive and return it for V7/V8 fusion analysis.  Do not upload
the experiment ZIP or result TAR to the competition platform.

Promotion requires weighted development gain at least `+0.35`, score gain at
least `+0.10` and mAP50 gain at least `+0.01` in both views, with recall loss no
worse than `0.003`.  Failure stops this branch.
