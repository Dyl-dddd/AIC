# 第三阶段：单模型权重插值

第二阶段证明旧 epoch25 与新 epoch5 存在互补，但双模型融合会增加推理成本，而且同 dev 路由有选参偏差。本阶段不训练，生成三个仍可单独加载的 checkpoint：新权重占比 25%、50%、75%，随后在冻结 dev 的原图和裁块视图统一评估。

## 上传与预检

上传 `semifinal_stage3_soup_20260924.zip` 至 `/mnt/proj`：

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_stage3_soup_20260924.zip /mnt/proj
cd /mnt/proj/semifinal_stage3_20260924
nvidia-smi

/mnt/proj/iron/.venv/bin/python scripts/run_semifinal_stage3.py \
  --project-root /mnt/proj/iron --preflight-only
```

默认父权重：

- `/mnt/proj/iron/best_score_proxy_epoch25.pt`
- `/mnt/proj/iron/runs/semifinal/grid2x3_ft15_lr3e5_b2/weights/epoch5.pt`

如路径不同，在所有命令后同时添加：

```bash
--baseline /实际路径/best_score_proxy_epoch25.pt --candidate /实际路径/epoch5.pt
```

脚本强制校验两份父权重 SHA，并验证 499 个张量的名称、形状和类型一致。生成权重放在本包的 `generated_weights/`，不修改原权重。

## 冒烟与完整运行

```bash
set -o pipefail
/mnt/proj/iron/.venv/bin/python -u scripts/run_semifinal_stage3.py \
  --project-root /mnt/proj/iron --limit 2 --output smoke_results \
  2>&1 | tee smoke.log
```

```bash
set -o pipefail
/mnt/proj/iron/.venv/bin/python -u scripts/run_semifinal_stage3.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee stage3.log
```

预计约 12–25 分钟。每个单元有固定签名，重跑会跳过已完成单元。不要并行启动两份。

## 回传

```bash
cd /mnt/proj/semifinal_stage3_20260924
tar -czf /mnt/proj/semifinal_stage3_results_20260924.tar.gz results stage3.log
```

下载 `/mnt/proj/semifinal_stage3_results_20260924.tar.gz`。暂时不要删除 `generated_weights/`；如果某个插值权重通过双视图门槛，还需要单独下载对应 `.pt`。

