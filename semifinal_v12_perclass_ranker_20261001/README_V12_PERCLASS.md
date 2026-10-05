# V12 independent per-class ranker

V12 fits one strictly group-OOF-validated XGBoost ranker for each of the eight
submitted defect classes. It reuses the completed V10 VF/P1/P2 Test caches and
does not run GPU inference.

Run:

```bash
python -u generate_v12_perclass_submission.py \
  --project-root /mnt/proj/iron \
  --v10-root /mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930
```

Submit only:

```text
/mnt/proj/iron/runs/semifinal/v12_perclass_ranker_submission/semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip
```

The generated ZIP contains exactly one file named `submission.json`.
