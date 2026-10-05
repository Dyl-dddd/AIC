# V35 operating score audit and V36 proposal quality experiment

Date: 2026-10-04 (Beijing). Protected official result: V12 67.92.

## V35 measured result

The full score-threshold curve was evaluated using the existing V12 group-OOF predictions. Each original image receives weight 488/459 and each derived crop 300/2754. This reproduces the expected image-count mixture more faithfully than averaging view scores; it is still a development estimate, not official Test evaluation.

| Policy | Mixed development score | Precision | Recall | AP50 |
|---|---:|---:|---:|---:|
| V12 unchanged | 69.339204 | 0.004433 | 0.968566 | 0.556828 |
| Best scanned shared threshold 0.006155 | 69.363856 | 0.006570 | 0.968265 | 0.556828 |

Four-fold threshold selection evaluated on the excluded group fold produced score deltas +0.03935, -0.03995, -0.02714, +0.02515, +0.02053. This is not a stable or useful gain. No V35 Test submission was generated. Existing OOF ranker scores are used; the ranker itself was not nested-refitted for these threshold folds, so this is a diagnostic rather than a claim of unbiased end-to-end cross-validation.

The independent reviewer also checked earlier V18 results and confirmed that a small FP reduction yields negligible precision benefit at 165,421 predictions. One lost official TP costs approximately 60/894 = 0.0671 score before any AP change. OOF and deployed ranker score distributions differ, making absolute thresholds particularly risky.

## V36 predeclared design

Hypothesis: pixels surrounding a proposal and an explicit proposal mask can predict localization quality that is missing from the geometry/support-only ranker. This differs from V20's category classifier, which did not learn low-IoU versus well-localized proposals.

- Initialize the MobileNetV3-small backbone from V20 train-only weights.
- Fit only on official `train` images. V20's internal training-group holdout convention is retained. Those groups were already used for V20 checkpoint selection, so internal validation loss is an optimization diagnostic, not unbiased generalization evidence. The official dev remains separate from both networks' fitting.
- Generate original, shifted, expanded, shortened and wrong-class proposals with known IoU targets; add background proposals. No official dev or Test image enters fitting.
- Use a four-channel input: three grayscale copies with ImageNet normalization plus the proposal mask. Condition on class and proposal geometry.
- Train binary IoU>=0.5 quality and continuous IoU jointly. First pilot: three epochs, batch 128, workers 0.
- Keep existing proposal membership and coordinates. On independent V12 OOF dev, re-score at most the three highest-score proposals per image/class with `old_score * quality**alpha`, alpha in {0.1, 0.25, 0.5}.
- Initial all-class gate: both original and grid views must improve development score by more than 0.1. A passing exploratory result still requires group stability checks before any Test deployment. No claim of guaranteed official gain is made.

Code: `scripts/v36_box_quality.py`, `scripts/evaluate_v36_box_quality.py`.
Outputs: `runs/semifinal/v36_box_quality_20261004/`.

## Checks before the main run

A 40-image, one-epoch preparation/training smoke passed. Synthetic metric checks for one true detection plus one duplicate and for empty predictions passed. Source scripts compile. Main train-only preparation/training is in progress; official scoring is unverified.

The sole reviewer identified view-dependent geometry and incomplete cache keys. All 2395 source train images were verified to be 4096x3000; changing the geometry reference to fixed 4096x3000 therefore leaves the running training inputs identical and fixes grid-view inference. Cache keys now include source code, GT and source-image file size/mtime in addition to weights, candidates and top-k.

## V36 outcome and rejection

Three local GPU epochs completed on 40,071 synthetic proposals. Best internal loss fell from 0.62785 to 0.54432, but the frozen V12 detector-candidate audit failed. The conservative `alpha=0.1` policy changed original view score 69.32434 → 69.05386 (−0.27048), and grid view 69.81303 → 69.75437 (−0.05867). Original zonglie AP50 fell by 0.01979. Larger alpha values were worse. First-epoch weights also failed both views (−0.38553 and −0.27333 at alpha=0.1). Thus internal synthetic validation loss did not predict real candidate-ranking quality. No V36 Test submission or training package was generated. Evidence: `runs/semifinal/v36_box_quality_20261004/dev_report.json` and `pilot_epoch1/dev_report.json`.
