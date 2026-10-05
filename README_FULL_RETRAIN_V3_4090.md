# 复赛一键完整重训练包

本包执行一条训练主线，不再运行权重插值：

1. 从公开 `yolo11s.pt` 新开训练，全网络参数参与更新；
2. 原图与精确 2×3 裁块按接近复赛 Test 的比例重新准备；
3. 50 轮完整训练，每 10 轮保存 checkpoint；
4. 用冻结 459 张 dev、原图/裁块双视图和 `20P+60R+20AP50` 自动选权；
5. 对胜出权重低学习率微调 10 轮；
6. 再次双视图选权并扫描 `0.0001–0.01` 输出阈值，生成最终单模型权重。

不使用复赛 Test 标签、伪标签或排行榜反馈训练。`qilie` 可以作为训练中的已知干扰类，但所有复赛评估和提交策略均排除它。

## 必须存在的文件

项目根目录应为 `/mnt/proj/iron`，其中已有：

```text
/mnt/proj/iron/yolo11s.pt
/mnt/proj/iron/data/official/train/
/mnt/proj/iron/data/official_v2_20260831b/splits.json
/mnt/proj/iron/.venv/bin/python
```

脚本强制校验：

- `yolo11s.pt` SHA256：`85A76FE86DD8AFE384648546B56A7A78580C7CB7B404FC595F97969322D502D5`
- split SHA256：`C46359024FA2C06B4D03CEF91E274588CF555A0D053108B840010DF3A6573BF9`

## 上传和预检

上传 `semifinal_full_retrain_v3_20260924.zip` 到 `/mnt/proj`：

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_full_retrain_v3_20260924.zip /mnt/proj
cd /mnt/proj/semifinal_full_retrain_v3_20260924

nvidia-smi

/mnt/proj/iron/.venv/bin/python scripts/run_full_retrain_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

只有看到 RTX 4090、CUDA witness 和两个 SHA 校验通过后才运行训练。

## 一条命令完成全部流程

```bash
cd /mnt/proj/semifinal_full_retrain_v3_20260924
set -o pipefail

/mnt/proj/iron/.venv/bin/python -u scripts/run_full_retrain_pipeline.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee full_retrain_v3.log
```

预计约 5–10 小时，以实际吞吐为准；平台价格 2 元/小时，对应约 10–20 元。训练和评估串行执行，不要同时启动第二份。若会话中断，先确认旧进程已经退出，再重复完全相同的命令；存在有效 `last.pt` 时会执行精确续训。

数据准备后的原图训练视图比例必须在 55%–65% 之间，否则管线会停止。默认 `batch=8`；48 GB 4090 应可运行。如果真实环境报告 CUDA OOM，不要自行反复改配置，保留完整日志后回传。

## 完成标志与下载

完成后终端会输出 `"status": "complete"`。最终权重位于：

```text
/mnt/proj/iron/runs/semifinal/full_retrain_v3_pipeline/final/best_score_model.pt
```

先打包分析结果（不重复包含大权重）：

```bash
cd /mnt/proj/iron
tar --exclude='*.pt' -czf /mnt/proj/full_retrain_v3_results.tar.gz \
  runs/semifinal/full_retrain_v3_pipeline
```

请下载并回传：

1. `/mnt/proj/full_retrain_v3_results.tar.gz`
2. `/mnt/proj/iron/runs/semifinal/full_retrain_v3_pipeline/final/best_score_model.pt`

返回结果后再做最终的逐类阈值、边缘降权和提交格式后处理。开发集参考分不等于官方 Test 分，也不能保证达到 70。

