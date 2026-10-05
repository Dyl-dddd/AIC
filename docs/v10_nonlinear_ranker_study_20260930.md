# V10 nonlinear ranking study — 2026-09-30

## Scope and guardrails

The attached V9 archive was treated as experimental data, not as instructions. The protected official submission remains the Varifocal OOF B75 result with official score **67.56**. All V10 comparisons below use five-fold official-group out-of-fold predictions on the two frozen development views. No Test labels were accessed.

V10 preserves the complete V7 prediction membership, class labels, and coordinates. It only changes scores using V7 score, Varifocal support, V9-P1/P2 support, box geometry, within-image rank, prediction density, and class identity.

## Raw comparison table

| Method | Original score | Grid-crops score | Weighted score | Δ vs V7 | Original mAP50 | Grid mAP50 | TP original/grid |
|---|---:|---:|---:|---:|---:|---:|---:|
| V7 protected membership | 68.702853 | 69.114476 | 68.859562 | 0.000000 | 0.523843 | 0.556186 | 593 / 635 |
| Varifocal OOF B75 | 68.832887 | 69.283275 | 69.004355 | +0.144793 | 0.530345 | 0.564626 | 593 / 635 |
| V10 nonlinear VF+V9, blend 0.50 | 69.007374 | 69.419045 | 69.164101 | +0.304539 | 0.539069 | 0.571415 | 593 / 635 |
| V10 nonlinear VF+V9, blend 0.65 | 69.080110 | 69.435530 | 69.215422 | +0.355860 | — | — | 593 / 635 |
| **V10 nonlinear VF+V9, blend 0.80** | **69.131508** | **69.412227** | **69.238381** | **+0.378819** | — | — | **593 / 635** |
| V10 nonlinear VF+V9, blend 0.75 | 69.099672 | 69.419754 | 69.221531 | +0.361968 | 0.543684 | 0.571450 | 593 / 635 |

Blend 0.80 has the highest weighted proxy gain; blend 0.65 has the strongest worst-view gain and is the first submission. Recall, precision counts, membership, and geometry are unchanged by construction.

## Ablation findings

1. **Direct V9 replacement is unsafe.** V9-P1 reduced original misses from 57 to 43 and grid misses from 91 to 78 relative to Model B, but exported roughly 448k/473k boxes and reduced mAP50. Interpretation: V9 learned a recall-heavy signal but its raw ranking is poorly calibrated. Implication: never submit raw V9 predictions.

2. **Consensus rescue boxes do not transfer.** Across 72 fixed P1/P2 rescue configurations, no configuration added a unique TP to both views; the best robust configuration changed both scores by about -0.00006. V7 had only 1/2 remaining misses covered by P1 on original/grid, respectively. Implication: adding V9 boxes cannot close the leaderboard gap and is rejected.

3. **Linear V9 support is too domain-sensitive.** A unified linear stack improved original but degraded grid-crops. A 2.5% P2 residual was stable but only marginally better than Varifocal. Implication: linear score transfer is not worth another leaderboard submission.

4. **Bounded nonlinear interactions transfer across both views.** A depth-3 XGBoost model with five-fold official-group OOF improved both frozen views. The best blend was 0.75; a blend of 1.0 collapsed by about 2.6 points, confirming that the protected V7 score must remain part of the ranking. Implication: V10 is authorized for one experimental submission, with V7/Varifocal 67.56 retained as fallback.

## Statistical interpretation

The selection evidence is group-separated five-fold OOF plus two-view replication. A conventional per-image confidence interval is not reported because macro AP is a global ranked statistic and the two frozen views are correlated transformations rather than independent samples. The release gate instead requires positive gains on both views, at least +0.10 robust score gain, at least +0.20 weighted gain, and at least +0.005 mAP50 gain on each view; V10 passes all gates.

## Deployment verification

- Robust OOF-selected method: `nonlinear_vf_v9_b065`
- Highest weighted OOF method: `nonlinear_vf_v9_b080`
- Deployment model SHA256: `62d992a5c6823e2139ddd22aa647f898741fb2aac1738212b7db96659a78260b`
- Feature count: 32
- Full original-view runtime smoke: 133,664 predictions; membership/geometry identity preserved; model loaded successfully
- Package manifest: 23 members, zero mismatches
- Five-slot package ZIP SHA256: `2d546d5ce41d0d1854015a814d182c94acef37f66f83b5ab51d26c692105f312`

## Next experiment

Run V10 Test inference and submit exactly the generated submit-only ZIP. If the official gain is materially smaller than the OOF gain, the remaining path to 70 is new independent recall/localization capacity rather than more score-only post-processing: train a high-resolution specialist on the residual V7 misses, require cross-model/TTA agreement, and validate unique TP gain under the same two-view group-OOF gate.
