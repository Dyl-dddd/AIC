# 条带探针包 v1 交付说明（AIC 提升 V12 方向）

> 生成时间：2026-10-05 17:52 CST ·
> 生成方：SpikeBot 000 (CodeWhale-LOCAL) · 上游参考 Dyl-dddd/AIC

## 1. 这是什么

**在官方保护版本 V12（67.92）之上、追加"全高条带视图新增框"的探针提交包**
（= 上游 `docs/fullheight_strip_views.md` 规划的"最小版本：条带候选与主候选直接 NMS 合并"路线，
对应"从现有强权重直接做条带推理"——文档明说尚未被尝试过的路径）。

## 2. 数据依据（4090 实测，可复现）

**冻结 dev 459 图对照**（Model A 权重 `best_score_model.pt`，唯一变量 `--strip-pass`）：

- macro **AP50: 0.3459 → 0.3981（+5.2pt）**
- macro **Recall: 0.5940 → 0.6966（+10.3pt）**
- **zonglie（纵裂，V41 诊断的最大瓶颈）: AP50 +6.25pt、Recall +19.7pt**
- 9 类中 8 类 AP50 上涨；FP/图 6.24 → 7.61
- 原始证据：`runs/dev_off/metrics.json`、`runs/dev_on/metrics.json`（4090 上）

**官方计分公式核对**：score = 100×(0.2·P + 0.6·R + 0.2·AP50)
反推 V12=67.92（P 0.0053 / R 0.9754 / AP50 0.4646）完全吻合；
到 69 需要 AP50 **+0.054**（上游 V41 审计口径）。

## 3. 探针内容

- **基础**：V12 原始 **165,421 框逐字保留**（文本级追加，程序断言校验过"原框不变"）
- **新增**：**2,612 框**（条带候选去掉与 V12 同类框 IoU≥0.5 重复后；每图每类上限 50）
  - zonglie **159**、yiwuyaru 516、jieba 569、mamianmakeng 557、yanghuatiepi 466、gunyin 124、jiaza 122、huashang 99
  - 新增框提交分 = **0.0005**（低分区，只求 recall 不污染 AP 高分段）
  - 顺带补入 1 张 V12 零预测图（`0001432626-Raw03-f_00004_1_2.jpg`）
- **校验**：`validate_large_submission.py` 通过 → `"valid": true`
  - 788 图预期 / 782 有预测（原 781）
  - 8 类白名单，**无 qilie**（平台会拒 qilie 整包）
  - submission_bytes: 21,502,910

## 4. 预期效果（诚实版，非承诺）

- **P 成本**：可忽略（分母 165k→168k，扣约 -0.02 分）
- **R**：每捞回 1 个 FN = **+0.067 分**（官方共 22 FN，上限 +1.48）
- **AP50**：方向为正（dev +5.2pt），但上游有 5 次"dev 正向、官方不涨"前科
  ——**注意**：那些都是"只改分数"的实验（V14/V19/V22/V31/V33），本包是**改框集合**，
  与它们不同类，但仍需官方回执验证。
- **区间估计：+0.0 ~ +0.8 分**（权重主要在 FN 捞回；不该指望单次提交到 69）

## 5. 使用方式

### 文件位置与校验

- 服务器：`/mnt/proj/iron_strip_exp/runs/submission_strip_v1.json`（及 `.gz`）
- 本机副本：`D:\CodeWhaleData\deliveries\aic_strip_probe_v1\submission_strip_v1.json.gz`
- **SHA256 (gz)：`2f3090db1127b9746d70471d56ac0ca1de4178d9edfb36dfc3efb62675048b08`**
- 框数：**168,033**（= 165,421 原始 + 2,612 新增）
- 平台校验回执：见上方第 3 节（valid: true）

1. 文件：`submission_strip_v1.json`（服务器 `/mnt/proj/iron_strip_exp/runs/`）
   或压缩版 `submission_strip_v1.json.gz`（3.49MB，适合传输）
2. 提交前可自行再跑校验：
   ```bash
   python scripts/validate_large_submission.py submission_strip_v1.json \
     --source /mnt/proj/iron/data/semifinal/test \
     --report /tmp/validate.json --exclude-category qilie
   ```
3. **复赛"取最优"：提交零风险，永不覆盖 67.92**（上游文档口径）
4. 建议提交次数分配（今日剩约 3 次）：本探针 1 次 → 看官方回执再决定后续

## 6. 判读与后续

- **官方回执 R/AP50 上升** → 条带有效，可升级 v2（更大胆参数：中等分追加、放宽数量、或多视图）
- **无变化** → 提示 Test 域迁移性差，停止此方向（与上游 5 次失败同类结论）
- **下降** →（理论上不应发生；P 成本已算到可忽略）立即回退 V12 保护版

## 7. 复现材料

- 全链路命令与脚本：`D:\CodeWhaleData\tmp\aic_strip_results\`（merge_strip_probe.py、run_on_4090.py、EXPERIMENT_LOG.md）
- 4090 实验目录：`/mnt/proj/iron_strip_exp/`（含 dev 对照原始产物）
- Mojito 4060 早期对照：`D:\aic-strip-exp\`（已废弃，仅存档）
