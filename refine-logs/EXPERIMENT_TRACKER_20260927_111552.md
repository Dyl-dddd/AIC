# Ensemble V5 experiment tracker

Version: 20260927_111552

| ID | Block | Status | Acceptance gate |
|---|---|---|---|
| EV4-001..007 | YOLO11m training, selection, return archive | COMPLETE | Selected Model B and auditable result archive returned |
| EV4-008 | Model A+B frozen-dev calibration | COMPLETE | Global fusion beats Model A on both views |
| EV5-001 | Code-only inference package | COMPLETE | Isolated test, member manifest, and SHA256 pass |
| EV5-002 | Cloud preflight | PENDING_USER_GPU | 788 images, RTX 4090, both weight hashes valid |
| EV5-003 | Test cache and submissions | PENDING_USER_GPU | Two complete caches, two validated ZIPs, summary JSON |
| EV5-004 | Rule confirmation | PENDING_USER | Organizer/rule PDF permits multi-model inference |
| EV5-005 | Global fusion official submission | BLOCKED_ON_EV5-003_004 | Official score returned; retain only if above 66.16 |
| EV5-006 | Class router submission | BLOCKED_ON_EV5-005 | Global fusion shows positive transfer and submission budget remains |

Protected official baseline: 66.16. Primary frozen-development proxy: 68.297. The proxy is not an official score.

