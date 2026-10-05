# V12 bottleneck diagnosis and no-go decisions — 2026-10-04

## Protected result and objective

- Official best V12: 67.92; precision about 0.0053, recall 0.9754, AP50 0.4646, TP 872, FP 164549, FN 22.
- Official score: `20 * precision + 60 * recall + 20 * AP50`. Holding recall and precision constant, reaching 69.00 needs AP50 +0.054. Removing low-score FP alone has very small direct precision leverage; score-ranked AP or true localization/recall must improve.
- Preserve `D:/新建文件夹 (8)/semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip`, SHA256 `23C8D7F5BF89BC5A8E6C803D36ECAF2BF14893E2198594FF4F2D52B62BA26A49`. The experiments here do not modify it or produce a purportedly better submission.

## Observations (raw evidence)

1. V12 out-of-fold development predictions show a **localization bottleneck in zonglie**, especially full images. Original-view GT 61, AP30 0.6017, AP50 0.2591, AP75 0.0401; top-100 ranked TP changes 45 → 26 → 9 across these thresholds. Grid-view GT 82, AP30 0.6504, AP50 0.5464, AP75 0.1400. This is not the official Test AP, but the IoU sensitivity and inspected red/green examples establish that many apparent detections have inaccurate vertical extent.
2. Test geometry contains 488 full 4096×3000 images and 300 1387×1516 crops. The all-Test crop median grayscale standard deviation is 6.79, versus 9.19 for 200 sampled numeric train crops (KS=0.157, p=0.0051). Full-image contrast is similar (Test 20.04 vs train 20.44; KS=0.042, p=0.954). This is a covariate difference, not proof of lower Test detection accuracy.
3. A separate frozen numeric development crop set (1,757 crops from 295 source images) is also somewhat brighter/more contrasty than Test crops: median intensity 57.25 vs 50.39; median std 7.69 vs 6.79. A six-feature, one-to-one Test-matched 300-crop subset gets std 6.62 and intensity 53.47, but has only **5 zonglie GT** from 180 source images; its zonglie AP is too unstable for tuning. It has 37 mamian GT; V12 mamian AP50 is 0.787 versus 0.613 on all numeric dev crops. Thus low contrast alone does **not** explain the mamian Test hypothesis.
4. Unlabeled Test V12 output has 165,421 rows. Mamian outputs are highly concentrated: the ten most prolific full images, all in an adjacent `0001441825`–`0001441829` sequence, contain 22,093 of the 44,623 mamian rows (49.5%). This is a confirmed **prediction distribution anomaly**, not a verified official false-positive list. Even granting all 894 official GT to these images, at least 21,199 of those 22,093 boxes cannot be matched TPs; however, indiscriminate burst penalties also demote true defects (see below).
5. `qilie` is missing from V12 predictions despite being a ninth official class. Frozen dev has 10 q GT across two source groups. Source detector AP50 is 0.0078 and specialist attempts V37–V38 failed. Eight dev q boxes are much wider than the training q boxes. Test q prevalence and any contribution to the 22 official FN are unknown.

## Interventions tested, with their decision

| Intervention | Local evidence | Decision |
|---|---|---|
| Uniform zonglie box height increase / width shrink (V31) | Numeric dev zonglie AP improved, but official score 67.92 → 67.87 | Reject fixed geometry correction. |
| V34 fine-tuned detector as complement | Same numeric 295-image probe: V12 zonglie AP50 0.2329 (10/13 GT), V34 0.1457 (9/13); only one V34-unique GT at score ≥0.05 | Reject direct merge/replacement. |
| V42 numeric-domain weighted XGBoost V12 reranker | Zonglie grid numeric AP −0.0479; original numeric −0.0014; no robust class gain | Reject. |
| V44 penalize images with many mamian predictions | Numeric dev AP50 original 0.614→0.467 and grid 0.621→0.410 at mild exponent 0.25; Test-matched grid 0.787→0.592 | Reject image-count penalty. |
| V37–V38 q-specialist | Source AP50 0.0078; later checkpoint q AP <=0.0013 while emitting 77k–366k candidates | Reject. |

## Interpretation, limits, and next valid experiment

- Most defensible weakness: **zonglie geometry/ranking** plus a possible rare-class q omission. A second concern is repeated-texture images causing Test mamian candidate proliferation; their actual high-rank error rate cannot be measured without Test labels. Current development proxies are too small in relevant strata and have repeatedly disagreed with leaderboard feedback.
- Do **not** train on Test pseudo-labels or tune to unlabeled Test counts as if they were ground truth. Do not promise any increase to 68.57/69 based on these analyses.
- The next worthwhile model experiment is a **real-proposal** zonglie box-quality/localization head trained on source-group-separated training proposals and full-size image context, not synthetic shifted boxes (V36 failed). Gate it on untouched numeric and C dev groups: AP50 must improve in original **and** grid views without reducing matched recall; also require no material loss across other classes after V12 integration. If those gates fail, submit the protected V12.
- A complementary data action is a manual annotation audit of the widest q examples and representative high-texture negative/positive numeric rolls. The current q sample (two dev source groups) is too small to validate a reliable q specialist before deadline.

## Reproduction artifacts

- `scripts/audit_v41_v12_bottlenecks.py` → `runs/semifinal/v41_v12_bottleneck_audit_20261004/results.json`
- `scripts/probe_v41_v34_zonglie.py` → `runs/semifinal/v41_v34_probe_20261004/numeric_dev_report.json`
- `scripts/experiment_v42_numeric_weighted_ranker.py` → `runs/semifinal/v42_numeric_weighted_ranker_20261004/results.json`
- `scripts/audit_v43_test_matched_dev.py` → `runs/semifinal/v43_test_matched_dev_20261004/report.json`
- `scripts/experiment_v44_burst_rank_penalty.py` → `runs/semifinal/v44_burst_rank_penalty_20261004/results.json`

All numeric Test distribution observations are label-free; no official score uplift has been verified for the current experiments.
