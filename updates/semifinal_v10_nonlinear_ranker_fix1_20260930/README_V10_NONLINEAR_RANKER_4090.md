# V10 非线性候选框排序器（RTX 4090）

本包不会改变 V7 提交的候选框数量、类别或坐标，只使用 Varifocal、V9-P1、V9-P2 的支持证据重新排序 `score`。

冻结开发集五折 OOF 结果（不是官方榜分保证）：

- original：相对 V7 `+0.396819`
- grid-crops：相对 V7 `+0.305277`
- 加权：相对 V7 `+0.361968`

## 解压与运行

```bash
cd /mnt/proj/iron
unzip -qo semifinal_v10_nonlinear_ranker_fix1_20260930.zip
cd semifinal_v10_nonlinear_ranker_fix1_20260930
pip install xgboost==3.4.1
python -u scripts/run_v10_nonlinear_ranker_pipeline.py --project-root /mnt/proj/iron
```

程序会复用以下已有结果；若 Varifocal Test 预测不存在，会自动运行已有 Varifocal 推理：

- `/mnt/proj/iron/runs/semifinal/ensemble_v7_tta_submission/primary/submission.json`
- `/mnt/proj/iron/runs/semifinal/varifocal_oof_b75_submission/varifocal_test_predictions.json`
- `/mnt/proj/iron/runs/semifinal/v9_model_b_quality_p1/weights/last.pt`
- `/mnt/proj/iron/runs/semifinal/v9_model_b_quality_p2/weights/last.pt`

P1/P2 Test 推理可断点复用：只有完整预测和审计文件同时存在且哈希匹配时才会复用，临时文件不会被误当作完成结果。

## 最终只下载并提交这一个文件

```text
/mnt/proj/iron/runs/semifinal/v10_nonlinear_ranker_submission/semifinal_v10_nonlinear_ranker_SUBMIT_ONLY.zip
```

提交压缩包内只含一个 `submission.json`，避免平台出现 “Found multiple JSON files” 错误。

## 说明

开发代理的提升已经通过官方组五折 OOF 和两个冻结视图复验，但不能保证官方分数达到 70。当前官方 67.56 提交仍应保留，V10 作为下一次实验提交。
