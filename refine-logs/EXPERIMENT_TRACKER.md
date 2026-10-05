# Semifinal Ensemble V7 Experiment Tracker

当前截止日前跟踪表：`EXPERIMENT_TRACKER_20261003_2115.md`。下列旧 V7 状态保留为历史记录，不代表当前待执行任务。

Version: 20260927_TTA_RETURN

| ID | Block | Status | Acceptance gate | Output |
|---|---|---|---|---|
| EV4-001 | Package preflight | COMPLETE | Platform and package checks passed | `preflight.json` |
| EV4-002 | Balanced-v3 data | COMPLETE | Frozen split and dataset recipe verified | dataset summary |
| EV4-003 | YOLO11m phase 1 | COMPLETE | Training completed | phase-1 weights and logs |
| EV4-004 | Phase-1 checkpoint selection | COMPLETE | Frozen-dev evaluation completed | selection JSON and summary |
| EV4-005 | YOLO11m phase 2 | COMPLETE | Low-LR phase completed | phase-2 weights and logs |
| EV4-006 | Final checkpoint selection | COMPLETE | Model B selected and hashed | `model_b_yolo11m_best.pt` |
| EV4-007 | Compact result archive | COMPLETE | Returned archive audited | result TAR.GZ |
| EV4-008 | Model A+B calibration | COMPLETE | Global fusion beats Model A on both frozen views | 68.297 proxy |
| EV5-001 | Code-only inference package | COMPLETE | Isolated test and member manifest pass | `semifinal_ensemble_inference_v5_20260927.zip` |
| EV5-002 | Cloud preflight | COMPLETE_BY_PIPELINE_OUTPUT | Global output was produced by the guarded pipeline | printed preflight JSON |
| EV5-003 | Build Test candidates | GLOBAL_RETURN_VALIDATED | Global ZIP passes local structure and value audit; summary/routed artifact not returned | global ZIP SHA256 `F1BC5B7B...35B166F` |
| EV5-004 | Confirm ensemble acceptance | COMPLETE_BY_SCORE | Platform scored the multi-model artifact | official result |
| EV5-005 | Submit global fusion | COMPLETE | Official result 66.96, improvement +0.80 | leaderboard result |
| EV5-006 | Submit class router | CANCELLED | Same-dev gain only +0.112 and weaker worst view | no submission |
| EV6-001 | Direct final-JSON postprocess | COMPLETE_NEGATIVE | Extra NMS, voting, WBF screen, and global floors fail dual-view gate | direct postprocess report |
| EV6-002 | Source-aware consensus calibration | COMPLETE | Both views improve with unchanged TP | 68.367 proxy |
| EV6-003 | V6 code-only package | COMPLETE | Four isolated tests, manifest, and entry-point compile pass | V6 ZIP |
| EV6-004 | Build V6 consensus Test ZIP | PENDING_USER_CLOUD | Existing A/B caches validate and output ZIP passes audit | consensus ZIP + summary |
| EV6-005 | Model A/B native TTA full dev | COMPLETE | Four complete cells, both hashes and split fixed | returned TTA dev result TAR.GZ |
| EV6-006 | TTA fusion selection | COMPLETE | All six bounded configs improve both frozen views; best +0.493 | TTA fusion decision report |
| EV6-007 | TTA Test inference release | COMPLETE | Selected TTA sources only; isolated verification passes | V7 code-only ZIP |
| EV7-001 | V7 cloud preflight | PENDING_USER_GPU | RTX 4090, 788 images, protected hashes, TTA=true | printed preflight JSON |
| EV7-002 | Model A/B Test TTA caches | PENDING_EV7-001 | Exact policy and 788 records per cache | reusable cache pair |
| EV7-003 | Primary audited submission | PENDING_EV7-002 | Schema/image/class validation and ZIP integrity pass | V7 primary ZIP |
| EV7-004 | Primary official score | PENDING_USER_SUBMISSION | Compare against protected 66.96 | leaderboard result |
| EV7-005 | Conservative fallback | HOLD | Only consider after primary result and with quota remaining | V7 conservative ZIP |

Protected baseline: official score 66.96 from Model A+B global fusion.

Frozen-dev evidence: non-TTA V6 consensus 68.367; selected dual-native-TTA consensus 68.860 (+0.493), with original/grid scores 68.703/69.114. These are proxies, not official scores.
