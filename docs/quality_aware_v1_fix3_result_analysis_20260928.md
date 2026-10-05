# Quality-aware V1 fix3 result analysis

## Integrity and experiment contract

- Input archive SHA256: `3852e7e901d787403a2a973ee0e55a19c6f8d0e29d3b70b7849000151d671844`
- Result manifest: 57/57 files present, all SHA256 values match, no extra files.
- Both arms: 8 epochs, 1,667 optimizer updates, 12,304 microbatches.
- Initialization: 499/499 state tensors and 9,431,275/9,431,275 parameter elements match.
- Dataset/code signatures match; only the loss and run name differ.
- Final and Test data were not used.

## Raw results

| Arm | View | Score | mAP50 | AP75 | mAP50:95 | Recall | Precision | FP/image |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Protected Model A | Original | 66.4980 | .5259 | .2923 | .2990 | .9281 | .01473 | 82.78 |
| BCE continuation | Original | 66.5912 | .5373 | .3017 | .3054 | .9248 | .01780 | 68.04 |
| Varifocal | Original | 67.3840 | .5232 | .2914 | .2996 | .9477 | .00282 | 446.78 |
| Protected Model A | Grid | 65.5598 | .5639 | .2897 | .3139 | .8997 | .01503 | 14.09 |
| BCE continuation | Grid | 65.6182 | .5729 | .2999 | .3182 | .8967 | .01802 | 11.68 |
| Varifocal | Grid | 66.5966 | .5504 | .2922 | .3105 | .9255 | .00282 | 78.21 |

Weighted dual-view scores:

| Arm | Weighted score | Delta vs BCE |
|---|---:|---:|
| Protected Model A | 66.1408 | -0.0800 |
| BCE continuation | 66.2208 | 0 |
| Varifocal | 67.0842 | +0.8634 |

## Findings

1. **Observation:** Varifocal gains 14 TP on original and 19 TP on grid over BCE, but FP/image rises by 6.57x and 6.70x. Both views lose mAP50, AP75, and mAP50:95.
   **Interpretation:** the loss improved low-confidence recall coverage but damaged confidence ranking/calibration.
   **Implication:** the checkpoint must not replace Model A and must not enter the existing V7 Stage 2 path.

2. **Observation:** official-group bootstrap estimates the IoU>=.5 coverage gain over BCE at +2.29 percentage points on original and +2.89 points on grid, with both 95% intervals above zero.
   **Interpretation:** the checkpoint contains real complementary signal despite failing as a standalone detector.
   **Implication:** retain it only as a proposal/rescue expert.

3. **Observation:** most extra covered targets have very low Varifocal confidence (median about .0013), while the sub-.001 region contains roughly 149k/160k false positives in original/grid.
   **Interpretation:** a single global threshold cannot separate the useful tail from noise.
   **Implication:** threshold tuning and raw union are rejected.

4. **Observation:** a fixed cross-view-consensus, one-box-per-source rule selected 459 candidates but recovered no additional V7 TP; score changed by -0.00030/-0.00028.
   **Interpretation:** original/grid agreement from the same model is not independent evidence.
   **Implication:** this simple rescue rule is also rejected.

## Decision

`STOP` is correct for whole-model replacement. Preserve the official 67.49 V7 result and do not submit either fix3 checkpoint directly.

## Next experiment

The next preregistered mechanism should be a membership-preserving score-transfer gate:

1. Keep the full V7 candidate membership and coordinates unchanged.
2. Match Varifocal proposals only to existing low-score V7/BCE anchors of the same class at IoU >= .70.
3. Use official-group five-fold OOF calibration; original and every crop from a source remain in the same fold.
4. Transfer score only—never add a raw Varifocal box and never change coordinates.
5. Require positive score and mAP changes in both views, weighted gain >= .10, and no per-class AP50 drop beyond .02.

This tests the supported hypothesis: Varifocal is useful as confidence evidence for existing low-score boxes, not as a standalone ranking system or an unrestricted source of new boxes.
