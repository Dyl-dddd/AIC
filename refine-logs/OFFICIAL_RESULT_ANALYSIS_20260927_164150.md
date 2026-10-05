# Ensemble V5 official result and V6 escalation

Version: 20260927_164150

## Raw official result

| Candidate | Score | Recall | Precision | mAP@0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Protected Model A TTA | 66.16 | not returned | not returned | not returned | not returned | not returned | not returned |
| Model A+B global fusion | 66.96 | 0.9698 | 0.0086 | 0.4302 | 867 | 100,210 | 27 |

The score reconstructs as `20*0.0086 + 60*0.9698 + 20*0.4302 = 66.964`.

At unchanged precision and recall, reaching 70 would require mAP@0.5 about 0.582. Even perfect recall adds at most 1.812 points; with perfect recall and unchanged precision, mAP@0.5 still needs about 0.491. Therefore the next stage must improve ranking/localization AP while preserving recall.

## Direct-final-JSON ablation

| Method | Original | Grid crops | Weighted | Decision |
|---|---:|---:|---:|---|
| Current global fusion | 68.346 | 68.216 | 68.297 | Baseline |
| Extra NMS 0.68 | 68.289 | 68.240 | 68.271 | Reject; original view falls |
| Extra NMS 0.65 | 68.242 | 68.306 | 68.266 | Reject; original view falls |
| Global floor 0.00005 | 68.162 | 67.496 | 67.908 | Reject; TP and AP fall |

No direct manipulation of the returned final JSON improved both frozen views. The final JSON has lost model-source identity, so further blind NMS/thresholding is not justified.

## Source-aware result

| Method | Original | Grid crops | Weighted | TP original/grid | FP original/grid |
|---|---:|---:|---:|---:|---:|
| Plain global fusion | 68.346 | 68.216 | 68.297 | 590/624 | 82,161/86,939 |
| V6 consensus | 68.418 | 68.283 | 68.367 | 590/624 | 81,268/86,079 |

V6 uses cross-model match IoU 0.70, consensus score factor 1.10, Model A anchor coordinates, and a 0.50 penalty on unmatched Model B boxes. It preserves development TP in both views and improves ranking/precision modestly. This is a bounded small-gain candidate, not a claim that the official score will reach 70.

## Next experiments

1. Build the V6 consensus Test submission from existing A/B caches; no GPU inference required.
2. Run complete native multi-scale/flip TTA development evaluation for current Models A and B at 1024/1280.
3. Compare A-TTA+B, A+B-TTA, dual-TTA, and plain/TTA multi-source fusion on both frozen views.
4. Run TTA Test inference only for a development configuration that improves both views; otherwise stop the TTA branch.

