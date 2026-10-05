# Ensemble V6 native-TTA result analysis

Date: 2026-09-27

Returned archive SHA256: `EBCFF25AD4EFC370A9D18EACE5D2AD0E75E56F438A860A82D6E5226832FB1238`

The archive is complete: RTX 4090, Ultralytics 8.3.169, Torch 2.5.1+cu124,
459 frozen development sources, 612 allowed original-view ground-truth boxes,
both protected weight hashes, and all four Model A/B x original/grid-crop cells.

## Raw single-model data

Scores use the same local proxy `20P_micro + 60R_micro + 20mAP50_macro`.

|System|Original|Grid crops|Weighted|TP original/grid|FP original/grid|
|---|---:|---:|---:|---:|---:|
|Model A native TTA|67.772|67.321|67.600|583/614|83,826/92,108|
|Model B native TTA|67.409|66.880|67.207|580/619|65,242/67,701|
|V6 non-TTA consensus reference|68.418|68.283|68.367|590/624|81,268/86,079|

Neither TTA model alone beats the protected V6 consensus. TTA is useful only
as a complementary multi-model signal.

## Bounded dual-TTA fusion data

The broad non-TTA study had already rejected larger consensus boosts and full
unmatched-B weight. This study therefore kept boost `1.10`, B scale `0.35`,
unmatched-B factor `0.50`, and final NMS `0.72`, changing only match IoU and
coordinate mode.

|Configuration|Original|Grid crops|Weighted|Delta vs V6|TP original/grid|FP original/grid|
|---|---:|---:|---:|---:|---:|---:|
|IoU 0.70, anchor|68.703|69.114|68.860|+0.493|593/635|133,071/143,659|
|IoU 0.60, vote|68.664|69.088|68.825|+0.459|593/635|126,965/138,345|
|IoU 0.70, vote|68.631|69.112|68.814|+0.448|593/635|133,254/143,842|
|IoU 0.60, anchor|68.614|69.090|68.795|+0.429|592/635|126,898/138,290|
|IoU 0.50, vote|68.629|68.923|68.741|+0.374|593/633|122,761/134,520|
|IoU 0.50, anchor|68.578|68.925|68.710|+0.344|592/633|122,887/134,692|

All six configurations beat V6 in both views. The selected primary is IoU
0.70 with anchor coordinates. IoU 0.50 with voting is retained as the
lower-count fallback.

## Key findings

1. **Observation:** primary dual-TTA adds 3 original-view and 11 grid-view true
   positives over V6, raising the weighted proxy by 0.493.
   **Interpretation:** native scale/flip views recover difficult boxes that the
   plain A/B pair jointly misses. **Implication:** Test TTA is allowed through
   the development gate. **Next step:** run the fixed policy on all 788 Test
   images with no further parameter tuning.
2. **Observation:** false positives increase substantially despite the score
   gain. **Interpretation:** the gain is recall-led, and the official Test set
   already has high recall. **Implication:** the official improvement may be
   smaller than +0.493 and is not guaranteed. **Next step:** submit the primary
   first and preserve 66.96 as the rollback baseline.
3. **Observation:** every bounded configuration is robust across both domain
   views. **Interpretation:** the signal is not an isolated single-view peak.
   **Implication:** IoU 0.70 anchor is a defensible fixed selection rather than
   an unconstrained leaderboard guess. **Next step:** use the conservative
   artifact only after the primary official result is known.

## Decision

Gate: `BUILD_TTA_TEST_PACKAGE`.

Release: `../updates/semifinal_ensemble_v7_tta_20260927.zip`, SHA256
`7B8A4DC903E77B5E2172B00853C794C93360E7A56F742825B691A80862F1FD05`.

The official score remains the only Test-set decision signal; no Test labels
or platform feedback were used in this calibration.
