# V13 高分辨率互补专家训练包（RTX 4090）

当前官方保护分数为 **67.92**。本包不会读取 Test、不会生成提交文件，只训练和冻结评估两个互补专家：

- `hires_recall_bce`：1536 输入、BCE、受限 mosaic/位移/缩放，目标是补充 V12 尚未覆盖的真阳性。
- `hires_localize_vfl`：1536 输入、Varifocal、强化 box/DFL、弱几何扰动，目标是改善 IoU 与候选质量排序。

两个分支都从已验证的 Model B（SHA-256 固定）独立初始化。运行结束后，把结果归档下载给我；下一步才会在冻结开发集上检验它们与 V12 的互补性，并生成新的提交候选。

## 1. 上传与解压

把 `semifinal_v13_hires_experts_20261001.zip` 上传到 `/mnt/proj/iron`，然后运行：

```bash
cd /mnt/proj/iron
unzip -qo semifinal_v13_hires_experts_20261001.zip
cd semifinal_v13_hires_experts_20261001
```

## 2. 安装依赖

云平台已有可用 CUDA PyTorch 时不要重装 `torch`。安装与当前代码验证过的版本：

```bash
python -m pip install --no-cache-dir \
  numpy==1.26.4 scipy==1.14.1 pandas==2.2.3 \
  scikit-learn==1.5.2 opencv-python==4.10.0.84 \
  PyYAML==6.0.2 Pillow==11.0.0 py-cpuinfo==9.0.0
python -m pip install --no-cache-dir --no-deps ultralytics==8.3.169
```

检查环境：

```bash
python -c "import torch,cv2,numpy,pandas,sklearn,ultralytics; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0)); print('numpy',numpy.__version__,'cv2',cv2.__version__,'pandas',pandas.__version__,'sklearn',sklearn.__version__,'ultralytics',ultralytics.__version__)"
```

## 3. 只做预检

```bash
python -u scripts/run_v13_hires_experts_pipeline.py \
  --project-root /mnt/proj/iron \
  --preflight-only
```

预检会核对 RTX 4090、至少 18 GiB 空闲显存、Ultralytics 8.3.169、数据划分哈希和 Model B 权重哈希。任何一项不一致都会停止，避免浪费训练时间。

## 4. 一键训练、冻结验证与归档

```bash
python -u scripts/run_v13_hires_experts_pipeline.py \
  --project-root /mnt/proj/iron
```

预计约 **10–16 小时**，取决于存储和推理速度。两个训练分支顺序执行；重连后再次运行同一命令会复用已完成分支，并按现有训练审计逻辑从 `last.pt` 恢复未完成分支。

训练日志位于：

```text
/mnt/proj/iron/runs/semifinal/v13_hires_experts_pipeline/
```

## 5. 只下载这个结果包

```text
/mnt/proj/iron/semifinal_v13_hires_experts_results_20261001.tar.gz
```

不要把这个 TAR.GZ 提交到比赛平台。它包含两个 `last.pt`、双视图冻结开发预测、指标、训练审计和哈希清单，供下一阶段做 V12 融合与生成唯一 JSON 的提交 ZIP。

## 安全边界

- 不访问 Test，不消耗比赛提交次数。
- 不根据训练过程中的切片指标选 checkpoint；只比较预先指定的两个 `last.pt` 和旧 Model B。
- 即使新专家单独分数不高，也会保留其开发预测，因为独有 TP 和定位互补性才是其进入 V12 的主要价值。
- 67.92 始终作为保护回退；70 是目标，不能在官方评测前保证。
