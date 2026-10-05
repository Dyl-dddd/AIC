# Ensemble V5 global submission audit

Version: 20260927_152213

Source artifact: `D:/新建文件夹 (8)/semifinal_ensemble_v5_global_fusion.zip`

SHA256: `F1BC5B7BFAFC8D3D2FB99163F626EB70B4C703498B13CEB4319C154EE35B166F`

## Raw artifact table

| Field | Value |
|---|---:|
| ZIP members | `submission.json` only |
| ZIP CRC errors | 0 |
| Predictions | 101,077 |
| Images with predictions | 776 |
| Expected Test images | 788 |
| Images without predictions | 12 |
| Invalid records | 0 |
| Unknown categories | 0 |
| Excluded `qilie` predictions | 0 |
| Exact duplicate image/category/box keys | 0 |
| Minimum score | 0.000035 |
| Maximum score | 0.948731 |
| Maximum coordinate | 4096 |

| Category | Predictions |
|---|---:|
| `jieba` | 18,077 |
| `zonglie` | 15,081 |
| `jiaza` | 2,955 |
| `yiwuyaru` | 10,536 |
| `huashang` | 8,424 |
| `mamianmakeng` | 26,823 |
| `yanghuatiepi` | 15,044 |
| `gunyin` | 4,137 |

Per predicted image: mean 130.254, median 84, p95 370, p99 844, maximum 2,177 predictions.

## Decision

The artifact passes local structural and numerical validation. The 12 images without predictions are valid empty-prediction cases. The large prediction count and low score floor are consistent with the frozen recall-prioritized policy; only the official score can determine whether that policy transfers.

Submission is ready only after confirming that multi-model inference is permitted. Submit this global variant before the class-routed diagnostic. Preserve the official 66.16 baseline unless the returned score is higher.

