# Ensemble V4 result analysis

Version: 20260927_111552

## Audited inputs

- Manual archive SHA256: `CC5949F057214B3470514B7A1351CC02DA63DCA54AEE67EADBCD159DC2C9841F`.
- Result archive SHA256: `6A44AB512A51245F0781D95A9CE7E664EEE363D487D74428C40CBC9B24090A98`.
- Selected Model B: `model_b_yolo11m_best.pt`, SHA256 `02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090`.
- Protected Model A SHA256: `d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d`.

## Result

All scores below are frozen-development proxies under `20P + 60R + 20mAP50`, not official Test scores.

| System | Original | Grid crops | Weighted | Worst view |
|---|---:|---:|---:|---:|
| Model A | 67.137 | 65.831 | 66.640 | 65.831 |
| Model B | 65.779 | 62.280 | 64.447 | 62.280 |
| Global fusion | 68.346 | 68.216 | 68.297 | 68.216 |
| Class router diagnostic | 68.787 | 67.793 | 68.409 | 67.793 |

The bounded global fusion uses Model B score scale `0.35` and cross-model NMS IoU `0.72`. It improves Model A on both frozen views and is the primary candidate. The class router adds only `0.112` weighted proxy points and has a weaker worst view, so it remains a secondary, optimistic candidate.

## Decision

- Build and run the code-only Ensemble V5 Test inference package.
- Confirm that multi-model inference is allowed before any official submission.
- Submit the global fusion first. Keep the official 66.16 result unless a new official score is higher.
- Consider the class-routed candidate only after the global candidate returns a positive official result.
- Do not claim 70 from development proxies; use the next official result to measure the remaining domain gap.

Raw reproducible outputs are under `../runs/semifinal/ensemble_v4_analysis_20260927_v2/`.

