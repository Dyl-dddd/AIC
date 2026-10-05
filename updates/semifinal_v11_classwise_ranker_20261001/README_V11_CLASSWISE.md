# V11 class-adaptive ranker

This patch reuses the completed V10 P1/P2 test prediction caches. It performs
no training and no GPU inference. It creates three submit-only ZIP archives,
each containing exactly one `submission.json`.

Run from the extracted patch directory:

```bash
python -u generate_v11_classwise_submissions.py \
  --project-root /mnt/proj/iron \
  --v10-root /mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930
```

Submit `classwise_primary` first. Use `classwise_right_shift` only when the
primary official score improves over 67.77 by at least 0.08. Otherwise use
`classwise_shrink085` for the last submission.
