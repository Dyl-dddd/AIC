# V13 实验跟踪表

只有真实产物存在并通过对应门槛后才把状态改为 complete；开发指标不是官方 Test 分数。

| ID | 阶段 | 固定设置 | 主要证据 | 放行门槛 | 状态 |
|---|---|---|---|---|---|
| V13-001 | 环境预检 | RTX 4090；Ultralytics 8.3.169；Model B/split 固定哈希 | `preflight.json` | GPU、版本、显存、哈希和 CUDA witness 全通过 | pending |
| V13-002 | 召回专家 | YOLO11m；1536；BCE；12 ep；seed 20261001 | 训练日志、`initial_transfer.json`、`last.pt` | 完整训练且审计无退化 | pending |
| V13-003 | 定位专家 | YOLO11m；1536；VFL；12 ep；seed 20261002 | 训练日志、`initial_transfer.json`、`last.pt` | 完整训练且审计无退化 | pending |
| V13-004 | 冻结双视图评估 | 两个 last + 旧 Model B；阈值 0.0001 | `selection.json`、cells | 完整 459 源组、original/grid-crops 均完成 | pending |
| V13-005 | V12 融合审计 | 官方组 OOF；逐类排序；禁止 Test | OOF 报告、覆盖差集 | 加权 +0.20；双视图均正；recall 降幅 ≤0.002 | blocked-by-V13-004 |
| V13-006 | Test 候选 | 仅使用通过 V13-005 的固定方案 | 唯一 JSON、ZIP、清单和哈希 | 结构完整且保留 67.92 回退 | blocked-by-V13-005 |

## 当前保护结果

| 官方分数 | Recall | Precision | mAP50 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|
| 67.92 | 0.9754 | 0.0053 | 0.4646 | 872 | 164549 | 22 |
