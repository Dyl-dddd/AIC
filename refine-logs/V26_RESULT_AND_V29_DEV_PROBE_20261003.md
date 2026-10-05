# V26 result audit and V29 next measurement (2026-10-03)

V26 archive: `D:/新建文件夹 (8)/semifinal_v26_zonglie_context_results_20261003.tar.gz`. It contains `RESULT.json`, `README.md`, and `v26_zonglie_context_best.pt` (checkpoint SHA256 `cde6099278a041738b3ec56e619c3b469399d1401cbe4b12f3997658a4308abc`). No prediction files or competition submission are present.

| Result | Metric / value | Evaluation protocol |
| --- | ---: | --- |
| Protected V12 official score | 67.92 | competition Test |
| V26 mAP50 | 0.367902 | generated full/strip dev views |
| V26 zonglie AP50 | 0.119209 | generated full/strip dev views |
| V26 zonglie AP50-95 | 0.039306 | generated full/strip dev views |
| V26 qilie AP50 | 0.003473 | generated full/strip dev views |

The V26 dev view metric is not directly comparable with the official score or V12 original-view AP. In particular, the archived V26 result cannot establish a gain over 67.92 or show whether its true positives are new relative to V12. No direct submission or unvalidated score claims.

V29 probe ZIP: `D:/新建文件夹 (8)/semifinal_v29_v26_dev_probe_20261003.zip` (SHA256 `615ea13bc7581c2e0f3e9835cdfec6605af75f1a8cde4c7409bb5cc4c2a4db5e`). It uses the existing V26 best checkpoint on the cloud 4090 and emits same frozen dev images as the V12 original-view comparison. Per image it runs full-frame plus overlapping full-height strips, caches results and bundles only dev predictions. It excludes Test inference for a faster decision and preserves V12 as protected fallback.

On receiving the V29 result: compare V26 and V12 box AP50/AP75 per class under identical dev ground truth, count unique V26 zonglie true positives at IoU 0.5 and false positives, then test conservative candidate union and rank calibration. Advance to Test inference and submission packaging only if dev gains are robust under original and grid-crop views; otherwise reject V26 and keep V12.

Local RTX 4060 WDDM inference attempts at 1536, 1280, and 960/640 failed with GPU memory/driver errors. Do not use lower-resolution local runs as evidence of V26 quality. Package syntax and ZIP CRC passed.
