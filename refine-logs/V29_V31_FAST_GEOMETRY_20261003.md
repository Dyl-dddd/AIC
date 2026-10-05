# 2026-10-03 V29 result and V31 fast geometry probe

Protected official baseline remains V12 = 67.92. V29 result has exactly 459 dev images, 205,504 predictions, the V26 best-weight SHA256 `cde6099278a041738b3ec56e619c3b469399d1401cbe4b12f3997658a4308abc`, and the frozen split SHA256 `c46359024fa2c06b4d03cef91e274588cf555a0d053108b840010df3a6573bf9`. No Test labels were used.

## V26 versus V12 under the same original-view dev protocol

| Class / IoU 0.5 AP | V12 OOF | V26 | V26 unique GT at score >= 0.05 |
| --- | ---: | ---: | ---: |
| zonglie | 0.259114 | 0.122625 | 1 |
| yanghuatiepi | 0.360721 | 0.149962 | 3 |
| 8-class macro | 0.554918 | 0.292829 | — |

V26 zonglie GT coverage at score >= 0.05 was 36/61 versus V12 54/61; 19 truths were V12-only, just 1 V26-only. The best of six conservative box-union configurations improved zonglie AP50 by only 0.000097. Yanghuatiepi addition reduced AP50 in every tested configuration. V26 agreement boosting reduced zonglie AP50. **Decision: reject V26 for official submission and do not spend time on its Test inference.** This conclusion reflects same-image dev comparison, not an official Test score.

## V31 V12 zonglie box extent correction

The V25 error audit showed that top-ranked zonglie boxes were often too short vertically and somewhat wide. V31 tested a small, predeclared grid of height factors and width factors on frozen V12 OOF predictions, changing geometry only. A height factor of 1.5 and width factor of 0.85 was the strongest AP50 candidate in the numeric-source development subset, which is the only source type in the provided Test images.

| Numeric-source zonglie AP50 | V12 | V31 strong | Delta |
| --- | ---: | ---: | ---: |
| original/full images, 13 GT | 0.232901 | 0.468953 | +0.236051 |
| grid-crop images, 17 GT | 0.324608 | 0.371552 | +0.046944 |

Across all sources, original AP50 was 0.259114 -> 0.395560 and grid-crop AP50 0.546418 -> 0.553249. Caveats: numeric-source GT counts are small; AP75 **declined** on both views (numeric original 0.153813 -> 0.039604, grid 0.058766 -> 0.008664). The official platform reports mAP@0.5, but this decline means localization is not universally better and the hidden score is not assured to rise. Do not claim 70+.

Two single-root-JSON packages were built from the official 67.92 V12 package. They preserve 165,421 prediction rows, all scores, all classes, all image IDs, and all non-zonglie boxes. Only 27,142 zonglie boxes are geometry-adjusted in the strong package. All 788 Test image dimensions were checked to clip boxes safely. ZIP CRC and filename checks passed.

| Candidate | Geometry | SHA256 | Decision |
| --- | --- | --- | --- |
| `semifinal_v31_zonglie_extent_strong_SUBMIT_ONLY.zip` | h×1.5, w×0.85 | `fb895526905b8c7dd70de63e291e945969fc7656d91073fa021fa9a8d9ab9666` | first official probe if a slot remains |
| `semifinal_v31_zonglie_extent_conservative_SUBMIT_ONLY.zip` | h×1.25, w×1.0 | `98f5445e971b0d36d5f5647021a85cc77d37f68aa5e690b8e89691d45efd387e` | fallback only if slots remain; dev gain smaller |

If strong beats 67.92, retain it and inspect official mAP50/TP/FN before any next submission. If it falls, revert to V12; do not blindly submit the conservative variant. The score is not guaranteed. All output generation and raw dev metrics are reproducible via `scripts/experiment_v31_zonglie_box_extent.py`, `scripts/build_v31_extent_submission.py`, and `runs/semifinal/v31_zonglie_extent_20261003/`.

## Official follow-up (2026-10-03 17:54)

The strong V31 package scored **67.87**, down from protected V12 **67.92**. The official report had recall 0.9754, precision 0.0053, TP 872, FP 164549, and FN 22 unchanged; mAP50 fell to 0.4619. This is direct evidence that the large numeric-dev zonglie geometry gain did not transfer. **Reject both V31 variants and do not submit the conservative variant without a separate compelling reason.** This supersedes the prospective decision table above. V12 remains the known best.
