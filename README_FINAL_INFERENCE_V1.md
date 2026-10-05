# 复赛最终权重推理与提交包

本包不训练模型。它使用完整重训练选出的唯一权重：

```text
/mnt/proj/iron/runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt
SHA256 D69B236A91599F850B495D7B2E60D87C0C4905713398BACA21D18E564CC42A4D
```

一次 Test GPU 推理生成可复用候选缓存，再离线构建三份提交。所有提交都排除
复赛不允许的 `qilie`，并验证 788 张 Test 图片、类别、坐标、JSON 和 ZIP CRC。

提交格式是检测框数组；没有保留检测框的图片不会写入虚假占位框。验证报告中的
`missing_images` 仅表示这些图片没有预测，并不代表提交文件格式错误。

## 解压和预检

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_final_inference_v3_20260925.zip /mnt/proj
cd /mnt/proj/semifinal_final_inference_v3_20260925

/mnt/proj/iron/.venv/bin/python scripts/run_final_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

若 Test 实际不在 `/mnt/proj/iron/data/semifinal/Test`，在两条命令后补充：

```bash
--source /实际/Test/目录
```

预检必须显示 788 张图片和正确权重 SHA256。

## 正式生成

```bash
cd /mnt/proj/semifinal_final_inference_v3_20260925
set -o pipefail

/mnt/proj/iron/.venv/bin/python -u scripts/run_final_submission_pipeline.py \
  --project-root /mnt/proj/iron \
  2>&1 | tee final_inference_v1.log
```

输出目录：

```text
/mnt/proj/iron/runs/semifinal/final_submission_v1/
```

其中：

- `semifinal_dev_calibrated.zip`：逐类阈值版本，本地公式最高，冲分时优先提交；
- `semifinal_recall_safe.zip`：冻结开发集统一下限 `1e-4`，作为泛化备份；
- `semifinal_ultra_recall_1e5.zip`：极低阈值探索版，预测更多，只有提交次数允许时再测。

三个 ZIP 的根目录都只有一个 `submission.json`。完成后下载并回传：

```text
/mnt/proj/iron/runs/semifinal/final_submission_v1/FINAL_SUBMISSION_SUMMARY.json
```

优先上传 `semifinal_dev_calibrated.zip` 到竞赛平台。若提交次数允许，再用
`semifinal_recall_safe.zip` 做泛化对照。开发集代理分不等于官方 Test 分，
任何版本都不能保证达到 70 分。
