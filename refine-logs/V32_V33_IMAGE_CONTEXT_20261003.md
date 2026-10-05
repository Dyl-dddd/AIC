# 2026-10-03 image-context score reranking

Official formula supplied by user: `score = 20*Precision + 60*Recall + 20*AP50`.
V31 official: P=0.0053, R=0.9754, AP50=0.4619, score=67.87. R is the largest coefficient but only 22/894 GT remain undetected, so perfect recall at fixed P/AP50 adds at most 1.48 points. The V31 decline was in AP50 candidate ranking, not recall or precision.

Protected known-best remains V12=67.92. V31=67.87 is rejected.

## Design and isolation

Numeric-source official train only (1,457 images) fitted a fixed XGBoost image-context model from 64x64 grayscale texture features. Frozen numeric-source dev (295 images) was independent; test labels were not accessed. V12 OOF box predictions and annotations for the dev split evaluated class AP50. Each class and `any-defect` probability was tested at predeclared score multipliers `score * p^w` for w={0.25,0.5,1.0}. No boxes were changed or removed.

Only `mamianmakeng` improved on both original and grid-crop views: original AP50 0.61432 -> 0.67235 at w=0.25; grid using full-image context 0.62091 -> 0.67610. All other classes were rejected; some declined sharply. Full image presence model had dev image AP=0.25064 and ROC AUC=0.7290, over 33 positive dev images, so uncertainty remains.

To address the Test mix of full 4096x3000 images and 1387x1516 crop images, a second XGBoost model was fit on six 1387x1516 crops per training image and tested on 1,770 dev crops. Crop-level mamian presence AP=0.42826, ROC AUC=0.87524; 74 positive dev crops. Frozen V12 OOF crop-box AP50 0.62091 -> **0.66125** at conservative weight 0.1. Weight 1.0 was harmful (0.56153), so this must remain a light prior.

## V33 build

The package `D:\新建文件夹 (8)\semifinal_v33_mamian_context_rerank_SUBMIT_ONLY.zip` was built from the exact V12 submission (SHA256 `23c8d7f5bf89bc5a8e6c803d36ecaf2bf14893e2198594ff4f2d52b62ba26a49`). It has one root `submission.json`, exactly 165,421 rows, and SHA256 `09f107c270515613135601b1e44d5a9241da59498e5a06f848a293de20583ecf`. Exactly 44,623 mamianmakeng scores were changed; every image ID, class, bbox, other-class score, and row count was preserved. Test images were decoded only to compute 64x64 texture features; no Test labels were read. Full Test images use p^0.25; crop Test images use p^0.1. The ZIP CRC, JSON round-trip, and non-target identity were checked.

The weighted dev AP50 estimate from this single-class change is about +0.0064 macro AP50, equivalent to about +0.13 official score *if it transfers*. This is a hypothesis, not a guarantee; the V31 official reversal shows the domain gap. Submit only if a quota slot is available and keep V12 as fallback. No claim of reaching 70 is supported.

Reproduce with `scripts/experiment_v32_image_context_rerank.py`, `scripts/experiment_v33_crop_context_rerank.py`, and `scripts/build_v33_mamian_context_submission.py`. Reports are in `runs/semifinal/v32_image_context_20261003/` and `runs/semifinal/v33_crop_context_20261003/`.

## Official outcome and decision

The user submitted V33 on 2026-10-03. Official score was **67.88**, precision=0.0053, recall=0.9754, AP50=0.4623, TP=872, FP=164549, FN=22. This is below protected V12=67.92. The dev AP gain did not transfer; reject V33. No further single-class image-context reranking should be submitted without an independent, official-like check.
