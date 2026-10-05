# SAFE-R2 candidate ranking and localization study

Date: 2026-09-28

## Executive decision

The current protected official result remains **67.49**. None of the tested
SAFE-R2 variants is authorized for Test or leaderboard deployment.

The study found a repeatable but small ranking signal. The numerically strongest
group-out-of-fold diagnostic was the original-trained logistic reranker:

- original view: **+0.165178** score;
- grid-crop view: **+0.014428** score;
- weighted development score: **+0.107786**.

Its severe view asymmetry prevents release. The strongest model trained on
both views was a depth-two nonlinear quality reranker:

- original view: **+0.128533** score, **+0.0064266** mAP50;
- grid-crop view: **+0.023533** score, **+0.0011767** mAP50;
- weighted development score: **+0.088558**;
- TP, FP, FN, recall, box coordinates, and prediction membership unchanged.

This misses the preregistered release gate of +0.20 weighted score, +0.10
score and +0.005 mAP50 in each view. The result is useful evidence, not a
submission candidate.

Guarded localization refinement was negative in both views and is rejected.

## Motivation

The official V7 result improved from 66.96 to 67.49:

- recall: 0.9698 -> 0.9754;
- mAP50: 0.4302 -> 0.4433;
- TP: 867 -> 872;
- FP: 100,210 -> 164,549.

The gain is real but recall-led, while the large FP increase shows that the
next useful mechanism must improve ordering and localization quality rather
than add more low-confidence proposals.

SAFE-R2 therefore separates two questions:

1. **CAGE-Q**: can capped, source-aware geometric agreement estimate a box's
   probability of reaching IoU 0.5 and improve ranking?
2. **UG-RBV / DAGR**: can robust coordinate voting improve localization only
   when independent sources agree and uncertainty is low?

At most one representative is retained from each A/B x local/global bucket,
so repeated TTA or tile detections cannot manufacture independent support.

## Leakage controls

- Both frozen development views replay the exact V7 baseline:
  - original: 68.70285282028146;
  - grid crops: 69.11447644900377.
- Five folds are assigned by the official duplicate group, not by image or
  crop. A held-out group is excluded from both views.
- OOF predictions are concatenated before AP calculation; fold AP values are
  not averaged.
- Test and the 291-image final lockbox were never read by the analyzers.
- Reranking variants add or remove no prediction and change no coordinates.
- The lockbox runner is fail-closed while `deployment_allowed=false`.

## Results

| Family | Best variant | Original delta | Grid delta | Weighted delta | Decision |
|---|---|---:|---:|---:|---|
| Fixed heuristic | CAGE-Q | +0.119747 | -0.010879 | +0.070016 | STOP |
| Fixed view-invariant | CAGE-Q-VI | +0.052447 | -0.016826 | +0.026074 | STOP |
| Original-trained logistic OOF | full, blend 0.50 | +0.165178 | +0.014428 | +0.107786 | STOP |
| Cross-view logistic OOF | shared, blend 0.25 | +0.038327 | +0.012249 | +0.028399 | STOP |
| Cross-view shallow nonlinear OOF | HGB depth 2, blend 0.50 | +0.128533 | +0.023533 | +0.088558 | STOP |
| Guarded localization | UG-RBV only | -0.015908 | -0.004483 | -0.011559 | REJECT |

The fixed and learned experiments agree on the failure mode. Provenance has
different acquisition semantics in the two views:

- single-source predictions: 67.1% original vs 23.8% grid;
- four-source predictions: 4.6% original vs 20.2% grid;
- internal-edge candidates: 14,233 original vs 0 grid.

Increasing reranking strength improves original but eventually degrades grid.
This is a domain-shift ceiling, not a threshold-selection problem.

## Module decision

### Keep as research infrastructure

- provenance-capped evidence extraction;
- group-OOF evaluation and exact baseline assertions;
- membership hashes and lossless fallback;
- fail-closed final-lockbox runner;
- nonlinear diagnostic as evidence that quality ranking is learnable.

### Do not deploy

- fixed CAGE-Q on Test;
- any current learned SAFE-R2 profile;
- global/edge coordinate donors;
- UG-RBV/DAGR coordinate refinement;
- hard pruning or a second NMS.

## Next high-leverage direction

Post-processing is now empirically near its safe ceiling. Moving from 67.49
toward 70 requires a detector-level quality/localization change, not another
small score formula. The next experiment should train a quality-aware detector
or auxiliary IoU head whose confidence is supervised by matched IoU, using
official-group train/validation isolation and the same dual-view gate. Its
primary endpoints should be mAP50, AP75, and calibration of score versus IoU;
recall must remain protected by the existing V7 rescue lane.

Only after that model passes development OOF should the prepared final lockbox
be opened once. A STOP lockbox result is terminal for the frozen profile and
must not be used for retuning.

## Reproducible artifacts

- `steel_defect/safe_r2.py`
- `scripts/analyze_safe_r2_heuristic.py`
- `scripts/analyze_safe_r2_view_invariant.py`
- `scripts/analyze_safe_r2_crossfit.py`
- `scripts/analyze_safe_r2_crossview.py`
- `scripts/analyze_safe_r2_nonlinear.py`
- `scripts/run_safe_r2_final_lockbox_pipeline.py`
- `runs/semifinal/safe_r2_*/REPORT.md`
- `updates/safe_r2_final_lockbox_20260928.zip`

