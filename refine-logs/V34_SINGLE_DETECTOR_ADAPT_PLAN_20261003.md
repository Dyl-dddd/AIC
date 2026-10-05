# V34 single-detector adaptation: predeclared experiment

## Motivation

Best official V12 is 67.92. V31 geometry edits scored 67.87 and V33 image-context reranking scored 67.88 despite better frozen-dev AP. Both had unchanged official recall=0.9754 and precision=0.0053, so further blind score transforms have weak evidence. The official result has 164,549 false positives and only 22 false negatives. V26 from COCO-pretrained YOLO11m performed poorly on matched dev and must not be used as the starting weight.

## Hypothesis and intervention

Fine-tuning the stronger previously trained Model A single detector on a more faithful mixture of full images, bounded positive grid crops, all true-empty training images, and a small number of intact zonglie strips will improve localization ranking while avoiding V26's extreme oversampling of longitudinal strips. A bounded training-only low-contrast view tests sensitivity to the Test image texture and contrast. No Test image or pseudo-label enters training.

Package: `D:\新建文件夹 (8)\semifinal_v34_single_detector_adapt_fix1_20261003.zip`. The earlier unversioned ZIP was superseded before delivery; do not use it.
Starting weight SHA256: `d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d`.
Frozen split SHA256: `c46359024fa2c06b4d03cef91e274588cf555a0d053108b840010df3a6573bf9`.
Train 12 epochs at 1280 px, AdamW 6e-5, batch 4, workers 0, no mosaic/mixup/cutmix. Resume enforces workers 0.

## Comparison and gate

The source checkpoint and every saved candidate are validated under the **same** frozen-dev derived full+six-grid view dataset. Report macro AP50, AP50:95, precision, recall, and per-class AP50. Select the best *single* candidate checkpoint by dev AP50, but treat the official gain as unverified. Reject candidate if dev AP50 does not beat the source. Even if it does, retain V12 67.92 as protected fallback; a platform slot is justified only after a format-checked Test JSON has been produced. Never substitute this developer metric for the official score or claim 70 in advance.

## Operational constraints

The cloud machine must have the original Model A weight and official train/frozen split. Output TAR contains result report and selected detector weight, not a submission JSON. Submission building remains a separate audited step. The final method is a single detector, not simple voting/averaging of multiple networks or training stages. Competition result deadline is 2026-10-05 20:00 Beijing; code/assets deadline is 2026-10-07 23:59. Reserve time for Test inference and scoreboard submission after training.

## Local checks completed

`py_compile` of source scripts passed. Frozen split loaded and audited. A local 4-train/4-dev smoke data preparation completed, producing 8 train and 27 dev views with no changed-source or invalid-box errors. The upload ZIP passed CRC and isolated-import checks. Cloud GPU training and official Test score remain **not yet run/unknown**.
