# 优化计划产物清单

## 2026-09-24 第一阶段完整回传与第二阶段包

- `STAGE1_FULL_AUDIT_20260924.md`：对云端完整回传包、权重、split、缓存和评分的审计结论。
- `../runs/semifinal/stage1_postprocess_20260923/REPORT.md`：阈值、NMS、边缘降权的缓存复算报告。
- `../README_STAGE2_4090.md`：第二阶段4090执行说明。
- `../updates/semifinal_stage2_eval_20260924.zip`：低阈值与1280单变量复核包；58项测试通过，不含权重、数据和训练入口。

## 2026-09-24 第二阶段回传处理

- `STAGE2_RESULT_ANALYSIS_20260924.md`：6个评估单元的审计、单权重/融合结论与下一轮训练方向。
- `../runs/semifinal/stage2_analysis_20260924/REPORT.md`：原始数据表、逐类阈值与有界融合结果。
- `../runs/semifinal/stage2_analysis_20260924/audit.json`：上传指标逐框复算证据。
- `../runs/semifinal/stage2_analysis_20260924/class_thresholds.json`、`ensemble_diagnostics.json`：机器可读诊断。
- `../scripts/analyze_stage2_results.py`：可复现分析入口；当前测试 `72 passed`。

## 2026-09-24 第三阶段单模型权重插值包

- `../README_STAGE3_4090.md`：0.25/0.50/0.75 三个单模型插值 checkpoint 的4090执行说明。
- `../updates/semifinal_stage3_soup_20260924.zip`：代码包，不含权重、数据或训练入口；SHA256 `C34254AB1011400090FC00F90C9C83F812E177AB29A448057FC0D36D3CFD1F63`。
- `../scripts/build_semifinal_soups.py`、`run_semifinal_stage3.py`：父权重 SHA、张量兼容性、保存回读、冻结 dev 双视图评估。
- 真实两权重临时回读验证通过：499张量、9,461,340参数、9类；当前测试 `75 passed`。

## 2026-09-24 一键完整重训练包

- `../updates/semifinal_full_retrain_v3_20260924.zip`：从公开YOLO11s开始的全层训练、checkpoint双视图选权、低学习率微调和最终选权；SHA256 `EF7A7AEAB500F5FE7B011B0E2A1D4D52087DB13F5BB89E3DB7DBAF11EF0FEF58`。
- `../README_FULL_RETRAIN_V3_4090.md`：单命令运行、续训、预计时间和回传说明。
- `../configs/data/semifinal_full_retrain_v3.yaml`：原图三倍采样，预计训练原图占比约58.4%，匹配复赛约61.9%的原图构成。
- `../configs/train/semifinal_full_retrain_phase1_4090.yaml`、`semifinal_full_retrain_phase2_4090.yaml`：50轮全层训练和10轮低学习率微调。
- `../scripts/run_full_retrain_pipeline.py`、`select_full_retrain_checkpoint.py`：自动准备、精确续训、真实公式双视图选权和最终权重归档。
- 包内不含权重或图片；当前测试 `78 passed`，命令入口和ZIP成员哈希均已验证。

## 2026-09-23 计分规则确认及第一阶段代码

- SCORING_UPDATE_20260923.md：用户答疑公式20P+60R+20AP50、61.35复算、计划门槛修正。
- ../scripts/run_semifinal_stage1.py、../tests/test_stage1_geometry.py：独立云端评估与阈值对照，57项CPU测试通过，无模型训练。
- ../README_STAGE1_4090.md、../updates/semifinal_stage1_eval_20260923.zip：用户上传运行；SSH超时，云端GPU尚未验证。

## 2026-09-23 14:35 逼近70分计划

- `EXPERIMENT_PLAN_20260923_143511.md` → `EXPERIMENT_PLAN.md`：四个核心实验块，评分协议、几何适配、受控短训练、冻结与提交；目标不是分数保证。
- `EXPERIMENT_TRACKER_20260923_143511.md` → `EXPERIMENT_TRACKER.md`：历史证据与待执行项目分开登记。
- 上一版20260921时间戳计划/跟踪表保留；本轮只制定计划，没有启动训练、推理或平台提交。
- 技能共享输出协议文件缺失，采用时间戳版本、固定入口和本清单作为回退。

## 最新执行追加：2026-08-31 19:26

- EXPERIMENT_TRACKER_20260831_192600.md → EXPERIMENT_TRACKER.md：当前M1 stock五轮基线已真实训练。
- EXECUTION_20260831_192600.md：数据治理、48项测试、受控M0结果、M1命令和限制。
- stock低学习率诊断过关，P2诊断未过；当前只放行R020，不把全部计划标成完成。
- 实时状态：runs/m1_execution_20260831/state.json；历史18:45与16:55版本均保留。
- EXPERIMENT_PLAN.md / CODE_CHANGE_BACKLOG.md仅追加执行状态入口，原始16:55完整快照未改写。


## 执行追加：2026-08-31 18:45

- 用户已授权执行；此前“未启动训练”描述仅指16:55的计划起草轮。
- 当前执行清单：EXPERIMENT_TRACKER_20260831_184500.md → EXPERIMENT_TRACKER.md。
- 执行证据与限制：EXECUTION_20260831_184500.md。
- 生效数据清单：data/official_v2_20260831b/splits.json；样本复核：data/diagnostic_v2/review.json。
- R002已完成但学习失败；R002-FP32正在运行，后续依赖门槛。实时状态以runs/m0_execution_20260831_fp32/state.json为准。


创建：2026-08-31；版本：20260831_165522；语言：中文。

共享输出协议文件未安装，按 experiment-plan 主文件要求采用时间戳版本、固定入口和清单。未启动新训练，未修改原始图像/XML/ZIP/旧权重。

| 产物 | 时间戳版本 | 当前固定入口 | 类型 |
|---|---|---|---|
| 分阶段实验路线 | EXPERIMENT_PLAN_20260831_165522.md | EXPERIMENT_PLAN.md | 计划，不是实验结果 |
| 依赖与验收清单 | EXPERIMENT_TRACKER_20260831_165522.md | EXPERIMENT_TRACKER.md | 只有OLD-PILOT为完成实验 |
| 代码修改清单 | CODE_CHANGE_BACKLOG_20260831_165522.md | CODE_CHANGE_BACKLOG.md | 待实施工作 |

## 证据来源

- 本地旧实验：runs/steel/official_hybrid_local_2ep_20260831/results.csv、运行清单与stdout日志。
- 当前划分：data/official_yolo_hybrid_local/metadata/splits.json。
- 只读压缩包审计：对训练ZIP中CRC+大小相同的候选进行图像SHA-256复核；对train/test候选交叉验证；只汇总统计，不在计划中公开样本标识。
- 检出5组/10张训练图完全重复，3组跨原有划分；全部5组标注集合不同；train/test 1张内容重叠。候选C前缀跨划分89组，其物理身份含义尚未确认。
- 本地代码：voc.py、metrics.py、focal_trainer.py、train.py、eval.py、run_experiments.py、match_update_epochs.py，以及已安装Ultralytics 8.3.169的loss/trainer实现。
- 官网（2026-08-31复核）：https://www.aicomp.cn/tracks/tracks-6/4174.html 。

## 解释边界

- 内容泄漏影响评估可信度，不证明它导致低分。
- Focal软目标探针和预训练覆盖不足是待对照验证的风险，不是已确定故障原因。
- 时间按本机两轮76.30分钟外推；正式预算随v2数据量、批量及GPU实測重估。
# 2026-09-03 V3 冲榜计划

- `EXPERIMENT_PLAN_20260903_203000.md`：基于完整65轮结果与官网规则更新的主实验计划。
- `EXPERIMENT_TRACKER_20260903_203000.md`：带放行门槛、状态和算力预算的执行表。
- `../submissions/baseline_epoch5_20260903/README.md`：初赛基线提交包说明与哈希。

## 2026-09-21 复赛重训练计划

- `EXPERIMENT_PLAN_20260921_130000.md` → `EXPERIMENT_PLAN.md`：复赛混合尺寸与 2×3 裁块匹配训练计划。
- `EXPERIMENT_TRACKER_20260921_130000.md` → `EXPERIMENT_TRACKER.md`：SF001–SF008 执行跟踪。
- `semifinal_archive_audit_20260921.json`：复赛 ZIP 只读安全审计。
- `semifinal_domain_analysis_20260921.json`：训练/初赛/复赛聚合图像域统计。
- `../docs/semifinal_retraining_20260921.md`：4090 数据准备、训练、checkpoint 选择与推理命令。
- 共享输出协议文件未安装；采用时间戳版本、固定入口和本清单作为回退。

## 2026-09-26 Ensemble V4

- `EXPERIMENT_PLAN_20260926_093026.md` -> `EXPERIMENT_PLAN.md`: YOLO11m 1280 two-stage training and YOLO11s+YOLO11m ensemble plan.
- `EXPERIMENT_TRACKER_20260926_093026.md` -> `EXPERIMENT_TRACKER.md`: EV4-001 through EV4-008 execution gates.
- `../README_ENSEMBLE_TRAIN_V4_4090.md`: upload, preflight, resume, monitor, and result-download instructions.
- `../updates/semifinal_ensemble_train_v4_20260926.zip`: verified GPU training package; no dataset images or XML files.

## 2026-09-27 Ensemble V4 results and V5 inference

- `ENSEMBLE_V4_RESULT_ANALYSIS_20260927.md`: returned-archive audit, dual-view scores, decision gate, and interpretation boundary.
- `../runs/semifinal/ensemble_v4_analysis_20260927_v2/REPORT.md`: reproducible raw comparison and bounded fusion sweep.
- `EXPERIMENT_PLAN_20260927_111552.md` -> `EXPERIMENT_PLAN.md`: inference-only official validation plan.
- `EXPERIMENT_TRACKER_20260927_111552.md` -> `EXPERIMENT_TRACKER.md`: completed V4 gates and pending V5 user/GPU/rule gates.
- `../configs/inference/ensemble_v5_fusion.json`: frozen global-fusion and class-router profile.
- `../updates/semifinal_ensemble_inference_v5_20260927.zip`: code-only package; 23 members, no weights or images; SHA256 `E873B221511A721CC467CBC395AA6094473D3532938DB178920491995C958742`.
- Isolated package verification: two package-local tests passed, manifest verification passed, and all packaged Python entry points compiled. Repository-wide test collection remains affected by pre-existing duplicate module names under `delivery/`; the original four targeted current-source tests passed before the later local Windows page-file exhaustion, and the final package-local tests were rerun with unused heavy imports stubbed.
- `ENSEMBLE_V5_GLOBAL_SUBMISSION_AUDIT_20260927_152213.md`: returned global-fusion ZIP audit; 101,077 valid predictions, 776 images with predictions, eight permitted classes, no `qilie`, SHA256 `F1BC5B7BFAFC8D3D2FB99163F626EB70B4C703498B13CEB4319C154EE35B166F`.

## 2026-09-27 official 66.96 and Ensemble V6

- `OFFICIAL_RESULT_ANALYSIS_20260927_164150.md`: official score decomposition, direct-postprocess negative results, source-aware consensus gain, and TTA escalation gate.
- `../runs/semifinal/ensemble_v5_direct_postprocess_20260927_v2/REPORT.md`: extra NMS, box voting, and global-floor dual-view sweep; no direct final-JSON modification passes the robust gate.
- `../runs/semifinal/ensemble_v5_source_aware_20260927/REPORT.md`: A/B consensus ranking sweep; best weighted proxy 68.367 with both views improved and TP unchanged.
- `../configs/inference/ensemble_v6_consensus.json`: frozen source-aware consensus policy.
- `../updates/semifinal_ensemble_v6_gpu_20260927.zip`: code-only consensus and full TTA-development package; 26 members, no weights/images; SHA256 `DDAD428A364203E74278A880083DAD77A65D80CD94F7711F4A7BE8FCED0B6062`.
- Isolated V6 verification: four targeted tests passed, package manifest passed, all entry points compiled.

## 2026-09-27 returned TTA results and Ensemble V7

- `TTA_RESULT_ANALYSIS_20260927.md`: returned archive audit, raw four-cell table, bounded dual-TTA fusion table, findings, and Test gate.
- `../runs/semifinal/ensemble_v6_tta_fusion_bounded_20260927/REPORT.md`: reproducible six-configuration dual-view comparison; selected weighted proxy 68.860, +0.493 over V6.
- `../configs/inference/ensemble_v7_tta_primary.json`: fixed primary policy, match IoU 0.70 with anchor coordinates.
- `../configs/inference/ensemble_v7_tta_conservative.json`: lower-count fallback, match IoU 0.50 with coordinate voting.
- `../README_ENSEMBLE_V7_TTA_4090.md`: cloud preflight, resumable execution, output paths, and submission order.
- `../updates/semifinal_ensemble_v7_tta_20260927.zip`: code-only Test TTA package; 28 members, no weights/images; SHA256 `7B8A4DC903E77B5E2172B00853C794C93360E7A56F742825B691A80862F1FD05`.
- Isolated V7 verification: three targeted tests passed, all packaged Python files compiled, member count/root verified, and forbidden weight/image/XML members absent.

## 2026-10-01 V13 high-resolution experts

- `EXPERIMENT_PLAN_20261001_131353.md` -> `EXPERIMENT_PLAN.md`: two-arm high-resolution retraining, frozen evaluation, fusion gates, and stop rules.
- `EXPERIMENT_TRACKER_20261001_131353.md` -> `../EXPERIMENT_TRACKER.md`: V13-001 through V13-006 execution gates.
- `../README_V13_HIRES_EXPERTS_4090.md`: pinned environment, preflight, resumable run, and result-download instructions.
- `D:/新建文件夹 (8)/semifinal_v13_hires_experts_20261001.zip`: 28-member code/config package; no weights, images, XML, Test inference, or submission creation; SHA256 `0E37A4756CF958311F9849A519D5C84BEF4D0D1A2BA4664058AD036F6951EF1D`.
- Shared output-protocol files referenced by the experiment-plan skill were unavailable; timestamped artifacts, fixed entry points, and this manifest are used as the fallback.

## 2026-10-01 V13 results and V14 hybrid ranker

- `V13_RESULT_ANALYSIS_20261001.md`: archive integrity, training audit, raw metrics, V7 coverage difference, strict group-OOF ranking and release decision.
- `../runs/semifinal/v13_hires_experts_analysis_20261001/coverage.json`: reproducible GT-instance coverage audit; V13 VFL adds 1 original and 3 grid-crop hits beyond V7, while V13 BCE adds none.
- `../runs/semifinal/v14_v13_perclass_ranker_20261001/oof_results.json`: V13-evidence all-class OOF result; weighted delta vs V7 `+0.717337`, both views positive.
- `D:/新建文件夹 (8)/semifinal_v14_v13_hybrid_ranker_20261001.zip`: robust three-class V14 plus five-class V12 inference package; 14 members, isolated verification passed; SHA256 `C246FE0DD9A1E5B88B91353B1DF12CA701C2A167B5DE269CF686FB45D2961FBF`.

## 2026-10-02 V14 official result and V15 class attribution

- `OFFICIAL_V14_RESULT_ANALYSIS_20261002.md`: raw V12/V14 official comparison, ranking-only failure diagnosis, class-isolation ablation order, and stop rules.
- `D:/新建文件夹 (8)/semifinal_v15_class_attribution_20261002.zip`: standard-library-only V12/V14 class probe generator; four source members, no weights or data; SHA256 `FCCFD12EF51AB46B2A5EAE53485BC5239DD42C824F5AD5FFBBCE05B5750B91B4`.
- Isolated V15 verification: package manifest passed; three package-local tests passed; ZIP has one top-level package directory and excludes caches. Each generated submit-only ZIP is checked to contain exactly one root `submission.json`.

## 2026-10-02 V16 deadline ablation

- `V16_RESULT_ANALYSIS_20261002.md`: official-protected baseline, two-slot decision boundary, seven frozen-development ablation families, miss taxonomy, and stopping rule.
- `../runs/semifinal/v16_candidate_filter_20261002/results.json`: V12 OOF Top-K and unsupported-tail filtering, including per-instance misses.
- `../runs/semifinal/v16_geometry_rescue_20261002/results.json`: V13 expert coordinate transfer, rejected.
- `../runs/semifinal/v16_iou_ranker_20261002/results.json`: group-OOF localization-quality regression blend, rejected.
- `../runs/semifinal/v16_v10v12_blend_20261002/results.json`: group-OOF cross-ranker blend, rejected.
- `../runs/semifinal/v16_image_calibration_20261002/results.json`: image/class context and expert-support calibration, no material gain.
- `../runs/semifinal/v16_empty_image_rescue_20261002/results.json`: empty-image V13 fill, rejected.
- `../runs/semifinal/v16_box_dilation_20261002/results.json`: bounded class box dilation, no material gain.
- `D:/新建文件夹 (8)/semifinal_v16_mamia_jieba_SUBMIT_ONLY.zip`: format-validated, one-JSON experimental fallback; SHA256 `7DFD9160F6A72FECE3E85E374324C4CC72DD257260FAA05A335CB41B41A2880E`. No official result and not recommended as a route to 70.

## 2026-10-02 V17 targeted cloud package

- `../README_V17_TARGETED_RESCUE_4090.md`: prerequisites, upload commands, result artifact, and no-Test/no-submission boundary.
- `../scripts/run_v17_targeted_rescue_pipeline.py`: verified Model B initialization, focused train-only linked dataset, short training, frozen original/grid evaluation, and result archive.
- `../configs/train/semifinal_v17_targeted_rescue_4090.yaml`: one 1280-pixel 8-epoch expert; baseline V12 is not overwritten.
- `D:/新建文件夹 (8)/semifinal_v17_targeted_rescue_20261002.zip`: 25-member code-only package; extracted package tests passed 2/2 and CLI help smoke test passed; SHA256 `2E9BC397ACD5EE27CA031F99141B2529A319D159118A8F851A20C38F82DE7C0C`.

## 2026-10-02 V17 returned result audit

- `V17_RESULT_ANALYSIS_20261002.md`: archive integrity, frozen original/grid single-model comparison, unique-GT coverage, and reject decision.
- `../runs/semifinal/v17_targeted_rescue_analysis_20261002/coverage.json`: instance-level candidate coverage against the V7/V12 universe; V17 unique coverage 0 in both views.
- `../scripts/analyze_v17_targeted_rescue_coverage.py`: reproducible V17 coverage audit.
- `D:/新建文件夹 (8)/semifinal_v17_targeted_rescue_results_20261002.tar.gz`: received 26-member result archive; SHA256 `085BBD31E0A70DCCD25AB320746B714C7F836EDE92D399952BE6CB483C71C205`.

## 2026-10-02 V18 existing-result score-tail optimization

- `V18_SCORE_TAIL_ANALYSIS_20261002.md`: raw OOF table, TP boundary, package decision, and submission order.
- `../runs/semifinal/v18_score_tail_20261002/results.json`: group-OOF per-view/class score-tail exploration.
- `../runs/semifinal/v18_score_tail_20261002/results_mixed.json`: actual shared-threshold reproduction; pooled 10% is +0.009400/+0.010205 on original/grid with no dev TP loss, while pooled 18% loses one grid TP.
- `D:/新建文件夹 (8)/semifinal_v18_v12_tail10_SUBMIT_ONLY.zip`: V12-only 10% filter, single root JSON, 148879 valid predictions; SHA256 `DE48D60014258A43469AF95934DC0699FFAEBC2E1C9F09ED19174640D9A7E7DE`.
- `D:/新建文件夹 (8)/semifinal_v18_combo_tail10_SUBMIT_ONLY.zip`: two-class combo + 10% filter, single root JSON, 148879 valid predictions; SHA256 `53CD9962B4DD82ECE8AC1DAEC16DFB4F1FDC8DD53CBEB43E24CF73ADA5900790`.
- `D:/新建文件夹 (8)/semifinal_v18_combo_tail18_SUBMIT_ONLY.zip`: superseded/rejected 18% filter, single root JSON, 135645 valid predictions; pooled-dev recall loss means do not submit; SHA256 `AD487C9CC9F2D5EAA88AFAA477EA05F0D3643B0B1D7FAC9D90EE2F848B9B6D78`.

## 2026-10-02 V19 image-texture reranker

- `V19_TEXTURE_RERANKER_20261002.md`: method, corrected frozen group-OOF raw table, decision and no-guarantee caveat.
- `../scripts/analyze_v19_texture_reranker.py`: image-texture extraction and three-class strict group-OOF experiment.
- `../scripts/build_v19_texture_submission.py`: final class-specific training, V12 score-only rescore and single-JSON validation.
- `../runs/semifinal/v19_texture_reranker_20261002/results.json`: corrected OOF result; only yanghuatiepi accepted.
- `../runs/semifinal/v19_texture_submission_20261002/build_report.json`: source, output, identity, artifact hashes and target count.
- `D:/新建文件夹 (8)/semifinal_v19_texture_yanghuatiepi_v2_SUBMIT_ONLY.zip`: validated 165421-prediction single-JSON submission ZIP; SHA256 `AF7E475BF448F18E12B8394EC4B0B2DE49B970258864CB0A9006854E46A5A474`.
- `../runs/semifinal/v19_texture_submission_20261002/rejected_pre_crop_boundary.zip`: obsolete pre-correction archive, **do not submit**.

## 2026-10-03 deadline recovery plan

- `EXPERIMENT_PLAN_20261003_2115.md`: score/class audit, V34 exact-protocol replay, conditional hard-negative and zonglie single-detector fine-tunes, deadline gates.
- `EXPERIMENT_TRACKER_20261003_2115.md`: compact run IDs and current states; V34 is running on the user's cloud machine.
- `EXPERIMENT_PLAN.md` and `EXPERIMENT_TRACKER.md`: current-plan pointers; older experiments remain preserved as historical records.
