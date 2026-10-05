# V13 high-resolution expert result analysis

Date: 2026-10-01 (Asia/Shanghai)

## Integrity and scope

- Received archive: `semifinal_v13_hires_experts_results_20261001.tar.gz`
- SHA-256: `3186FA94F0A9328BE4913F72AF744C6C3C467DBBAFA2602D8749325A7EC3F17D`
- Result manifest: 38/38 members present and SHA-256 matched.
- Both arms completed 12 epochs with 2,491 optimizer updates, verified Model B initialization, RTX 4090, Torch 2.4.0 and Ultralytics 8.3.169.
- The archive did not access Test and did not create a submission.

## Raw frozen-development results

All rows use export threshold 0.0001. Scores are local proxies, not official leaderboard scores.

| model | view | predictions | TP | recall | precision | mAP50 | proxy score |
|---|---|---:|---:|---:|---:|---:|---:|
| Model B | original | 48,465 | 555 | 0.906863 | 0.011452 | 0.527870 | 65.198197 |
| V13 localize VFL | original | 762,443 | 576 | 0.941176 | 0.000755 | 0.514957 | 66.784839 |
| V13 recall BCE | original | 44,653 | 546 | 0.892157 | 0.012228 | 0.526456 | 64.303086 |
| Model B | grid-crops | 44,446 | 539 | 0.819149 | 0.012127 | 0.496547 | 59.322424 |
| V13 localize VFL | grid-crops | 711,844 | 577 | 0.876900 | 0.000811 | 0.488114 | 62.392465 |
| V13 recall BCE | grid-crops | 41,519 | 542 | 0.823708 | 0.013054 | 0.503485 | 59.753274 |

## Coverage against the protected V7/V12 candidate universe

Coverage means that at least one same-class prediction reaches IoU >= 0.50 for a frozen GT instance.

| view | expert | expert hits | unique vs V7 | V7-only | union hits | union recall |
|---|---|---:|---:|---:|---:|---:|
| original | V13 localize VFL | 576 | 1 | 18 | 594/612 | 0.970588 |
| original | V13 recall BCE | 547 | 0 | 46 | 593/612 | 0.968954 |
| grid-crops | V13 localize VFL | 577 | 3 | 61 | 638/658 | 0.969605 |
| grid-crops | V13 recall BCE | 543 | 0 | 92 | 635/658 | 0.965046 |

The VFL-only unique hits were one `zonglie` original instance and two `jieba` plus one `mamianmakeng` grid-crop instance. Three of four unique hits had score <=0.001188; directly adding all VFL boxes would import more than 700k predictions per view for at most four additional development hits.

## Strict group-OOF ranking results

V13 localization support was added as four features (`present`, IoU, log score and log advantage) to the existing independent per-class V12 rankers. Five-fold splits remained isolated by official duplicate group.

| ranker | original proxy | grid-crops proxy | weighted delta vs V7 | worst-view delta vs V7 |
|---|---:|---:|---:|---:|
| V7 candidate ranking | 68.702853 | 69.114476 | 0.000000 | 0.000000 |
| V12 official-tested ranker | 69.324337 | 69.813033 | +0.650826 | +0.621484 |
| V14 all-class V13 evidence | 69.337146 | 69.966897 | +0.717337 | +0.634294 |
| V14/V12 class hybrid | 69.398658 | 69.905329 | **+0.731991** | **+0.695806** |

The hybrid uses V14 models only for `jieba`, `jiaza` and `mamianmakeng`; the other five classes retain V12. Relative to V12 it improves original by +0.074322 and grid-crops by +0.092296 while preserving candidate membership and geometry.

## Key findings

1. **Observation:** V13 VFL raises standalone recall but creates 14–16 times more predictions than Model B. **Interpretation:** its score calibration is unsuitable for direct submission, but its overlap evidence contains ranking information. **Implication:** never submit V13 raw predictions. **Next step:** use only its matched support features.
2. **Observation:** the BCE recall arm adds no GT coverage beyond V7 on either view. **Interpretation:** it is mostly a lower-resolution-equivalent retraining outcome rather than a complementary detector. **Implication:** remove it from Test inference and preserve compute.
3. **Observation:** all-class V13 evidence improves both views over V12, but some classes regress. **Interpretation:** evidence value is class-dependent. **Implication:** the preregistered robust class hybrid is preferable to replacing every V12 model.
4. **Observation:** the hybrid improves both frozen views with unchanged boxes. **Interpretation:** the gain is candidate-ordering quality rather than candidate-count inflation. **Implication:** V14 is eligible for one official submission, while 67.92 remains the protected fallback.

## Next experiment

Run the V14 package once on Test, generate the submit-only ZIP, and use one official slot. Do not spend a slot on either raw V13 expert. If V14 transfers positively, subsequent work should target the remaining V7 misses by class-specific detector training rather than further global box inflation.

The proxy cannot guarantee a score of 70; only the official evaluator can establish the final score.
