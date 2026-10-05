# V17 定向候选专家：云端一键训练包

这是**训练与冻结验证包，不是比赛提交包**。它从已验证的 Model B 初始化，在原有 balanced-v3 训练集上为细小 `jieba`、细长 `zonglie` 和 `jiaza` 样本各增加一次训练视图；不改原始数据、不接触 Test。训练后在固定的 459 张开发图上同时评估原图与网格裁剪，并打包结果供后续互补性审计。V12 官方 67.92 保留不变；V17 没有分数保证。

## 云端前提

- `/mnt/proj/iron/runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt`
- `/mnt/proj/iron/data/official_v2_20260831b/splits.json`
- `/mnt/proj/iron/data/official/train/`
- `/mnt/proj/iron/data/semifinal_yolo_balanced_v3/steel_defect.yaml` 及其 `images/train`、`labels/train`、`images/dev`、`labels/dev`
- RTX 4090；已可导入匹配 CUDA 的 PyTorch 与 torchvision。预检会核对 Model B 和数据划分 SHA-256、Ultralytics 版本及 GPU。

## 上传、安装和运行

把 `semifinal_v17_targeted_rescue_20261002.zip` 上传到 `/mnt/proj/iron`。在云端终端执行：

```bash
cd /mnt/proj/iron
unzip -nq semifinal_v17_targeted_rescue_20261002.zip
cd semifinal_v17_targeted_rescue_20261002
python -m pip install --no-cache-dir -r requirements-v17.txt
python -u scripts/run_v17_targeted_rescue_pipeline.py --project-root /mnt/proj/iron --preflight-only
python -u scripts/run_v17_targeted_rescue_pipeline.py --project-root /mnt/proj/iron
```

若你正在使用 `(myenv)`，上述 `python` 就是该环境的 Python。不要执行 `pip install torch` 覆盖已能正常使用的 CUDA PyTorch。第一次完整运行将准备仅含链接的定向训练集；若中断，确认旧进程退出后重跑同一条命令，可复用完成的步骤。不要删除已生成的目录或 `last.pt`。

训练计划为单分支 YOLO11m、1280 输入、8 epoch、batch 6、AdamW 低学习率，之后对 Model B 与新 `last.pt` 做固定阈值双视图评估。预估约 5–9 小时，但以云端实际速度为准。若出现 OOM 或依赖错误，请保留完整报错，不要盲目改配置。

## 只下载并回传

```text
/mnt/proj/iron/semifinal_v17_targeted_rescue_results_20261002.tar.gz
```

**不要把结果 TAR.GZ 交到比赛平台。** 返回后先审计相对 V12 的独有 TP、两视图 AP/召回和提交风险；只有验证通过才会制作一个 `submission.json` 的提交 ZIP。当前 2 次官方提交机会不由此包消耗。
