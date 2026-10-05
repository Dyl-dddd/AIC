# 复赛一键完整重训练包（共享内存修复版）

此版本修复旧包训练时出现的：

```text
DataLoader worker is killed by signal: Bus error
```

根因是平台容器的 `/dev/shm` 共享内存不足，而旧配置使用了
`workers=2`。本包的数据准备和两个训练阶段均固定为 `workers=0`；数据由训练
主进程读取，不再创建会消耗共享内存的 DataLoader 子进程。GPU 前向、反向和
优化仍在 RTX 4090 上运行，`batch=8`、`imgsz=1024` 均未降低。

本包使用新的 `full_retrain_v3_fix1_*` 运行目录，刻意不恢复旧包产生的
`full_retrain_v3_phase1/weights/last.pt`，因为旧 checkpoint 可能保存并恢复
不安全的 `workers=2` 参数。旧目录无需删除。

## 必须存在的资源

```text
/mnt/proj/iron/yolo11s.pt
/mnt/proj/iron/data/official/train/
/mnt/proj/iron/data/official_v2_20260831b/splits.json
/mnt/proj/iron/.venv/bin/python
```

已经生成的 `/mnt/proj/iron/data/semifinal_yolo_balanced_v3` 会通过元数据校验后
直接复用，不会重复准备数据。

## 解压与预检

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_full_retrain_v3_fix1_20260925.zip /mnt/proj
cd /mnt/proj/semifinal_full_retrain_v3_fix1_20260925

nvidia-smi

/mnt/proj/iron/.venv/bin/python scripts/run_full_retrain_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

预检输出应包含 RTX 4090、`"dataloader_workers": 0`、CUDA witness、权重和
split 的 SHA256 校验结果。`shared_memory_gib` 很小不再影响本版本。

## 正式训练

```bash
cd /mnt/proj/semifinal_full_retrain_v3_fix1_20260925
set -o pipefail

/mnt/proj/iron/.venv/bin/python -u scripts/run_full_retrain_pipeline.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee full_retrain_v3_fix1.log
```

不要同时运行旧包或第二份训练。若终端断开，先用 `nvidia-smi` 确认没有旧训练
进程，再原样重跑上面的正式训练命令；修复版目录中的 `last.pt` 可以安全续训。

## 完成结果

完成时终端会输出 `"status": "complete"`。最终权重位于：

```text
/mnt/proj/iron/runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt
```

打包分析结果：

```bash
cd /mnt/proj/iron
tar --exclude='*.pt' -czf /mnt/proj/full_retrain_v3_fix1_results.tar.gz \
  runs/semifinal/full_retrain_v3_fix1_pipeline
```

请下载并回传：

1. `/mnt/proj/full_retrain_v3_fix1_results.tar.gz`
2. `/mnt/proj/iron/runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt`

开发集选权采用比赛公式 `20% Precision + 60% Recall + 20% mAP@0.5`。它用于选择
最可能有效的 checkpoint，但不能保证未知 Test 分数或保证达到 70 分。
