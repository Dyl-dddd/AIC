# 第二阶段：低生成阈值与 1280 分辨率对照

本包只做推理评估，不训练模型、不生成正式 Test 提交，也不修改 `/mnt/proj/iron`。第一阶段完整回传包已核验：RTX 4090、torch 2.5.1+cu124、Ultralytics 8.3.169、冻结 dev、两权重及发布代码哈希一致。

## 为什么运行这三组

第一阶段 8 组的最佳扫描点都落在生成阈值下界 0.001，已有缓存无法恢复更低置信度候选；因此补测 0.0001/0.0003。新 epoch5 搭配精确 2×3 网格是原图领先候选；1280 只对它做单变量对照。旧 epoch25+1024 滑窗保留为强基线。每次只加载一个权重，不做模型集成。

三组为：

1. 旧 epoch25，1024 滑窗；
2. 新 epoch5，1024 精确网格；
3. 新 epoch5，1280 精确网格。

三组都采用第一阶段缓存分析选出的 `edge_penalty=0.85`，NMS 维持 0.55；更改 NMS 到 0.35/0.70/0.85 均未提高参考分。每组分别评估 459 张原图和同源的 2754 个网格裁块，不把两者相加。

## 解压、预检、运行

上传 `semifinal_stage2_eval_20260924.zip` 至 `/mnt/proj`：

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_stage2_eval_20260924.zip /mnt/proj
cd /mnt/proj/semifinal_stage2_20260924
nvidia-smi
/mnt/proj/iron/.venv/bin/python scripts/run_semifinal_stage2.py \
  --project-root /mnt/proj/iron --preflight-only
```

默认权重与第一阶段相同：

- `/mnt/proj/iron/best_score_proxy_epoch25.pt`
- `/mnt/proj/iron/runs/semifinal/grid2x3_ft15_lr3e5_b2/weights/epoch5.pt`

若文件实际位置不同，在下面两条命令中同时添加 `--baseline /实际路径/旧权重.pt --candidate /实际路径/epoch5.pt`。脚本会校验固定 SHA，不能用其他同名文件替换。

先跑 2 张源图冒烟测试：

```bash
set -o pipefail
/mnt/proj/iron/.venv/bin/python -u scripts/run_semifinal_stage2.py \
  --project-root /mnt/proj/iron --limit 2 --output smoke_results \
  2>&1 | tee smoke.log
```

确认成功后运行完整对照：

```bash
set -o pipefail
/mnt/proj/iron/.venv/bin/python -u scripts/run_semifinal_stage2.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee stage2.log
```

预计约 15–30 分钟，按实际 GPU 状态为准。脚本串行运行，不要同时启动第二份。每个完整单元有签名，可在确认旧进程已经结束后重复同一命令，自动跳过已完成单元；单元内部不能续跑。

## 打包回传

```bash
cd /mnt/proj/semifinal_stage2_20260924
tar -czf /mnt/proj/semifinal_stage2_results_20260924.tar.gz results stage2.log
```

下载 `/mnt/proj/semifinal_stage2_results_20260924.tar.gz` 回传。保留整个 `results`，其中候选缓存用于后处理复算。

报告会对 0.0001、0.0003、0.001、0.003、0.01 分别重算同一输出的 Precision、Recall、mAP50 和 `20P+60R+20AP50` 本地参考分。它不是官方 Test 分，也不能保证达到 70。下一步是否训练，只根据本轮完整原图、裁块和逐类结果共同决定。
