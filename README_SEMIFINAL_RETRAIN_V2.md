# 复赛单模型重训练包（代码版，2026-09-23）

本 ZIP **不含训练数据、复赛 Test 或权重**。它在计算平台已有的 `/mnt/proj/iron` 项目上更新训练脚本和配置。不要把无标注复赛 Test、测试预测或伪标签用于训练。

## 这次与上一轮的区别

- 上轮 30 epoch、`lr0=1e-4` 的模型在裁块开发集有小幅收益，但独立作为单权重模型时原图开发集召回下降。本轮固定原有训练/开发划分，采用 15 epoch、`lr0=3e-5`、较轻几何和亮度增强，减少灾难性遗忘风险。**这是待验证的实验，不保证平台 70 分。**
- 预检核对 9 类训练标签顺序、train/val 非空及不重名、初始 epoch25 权重 SHA256、CUDA 可用性。训练数据仍来自 `data/official/train` 预处理结果；复赛提交需在推理后排除 `qilie`。
- 每 5 epoch 保存一次。训练结束后，选权脚本只比较**单个** checkpoint，并把原始 epoch25 作为基线。默认新模型在八个允许类别的开发集 mAP50 至少提升 0.01 且宏召回下降不超过 0.01 才通过；否则保留旧权重。这个门槛不是平台评分公式。

## 解压与训练

把 ZIP 上传到 `/mnt/proj/`，在计算平台终端执行：

```bash
cd /mnt/proj/iron
python3 -m zipfile -e /mnt/proj/semifinal_retrain_v2_code_only_20260923.zip /mnt/proj/iron
source .venv/bin/activate
python scripts/preflight_semifinal_retrain_v2.py
```

预检应显示 `ready: true`、`train_images: 7507`、`val_images: 1358`，且能识别 CUDA。若提示 `data/semifinal_yolo_grid_v1/steel_defect.yaml` 缺失，先确认旧数据是否仍在；不要对现有非空目录执行 `--overwrite`。若确实未准备过且已有原始训练数据与冻结 split，再运行：

```bash
python scripts/prepare_data.py --config configs/data/semifinal_grid_v1.yaml
python scripts/preflight_semifinal_retrain_v2.py
```

预检通过后开始训练：

```bash
mkdir -p logs
set -o pipefail
python -u scripts/train.py --config configs/train/semifinal_retrain_v2_4090.yaml 2>&1 | tee logs/semifinal_retrain_v2.log
```

只有训练命令成功退出，并且 `runs/semifinal/grid2x3_ft15_lr3e5_b2/completion.json` 的 `exit_code` 为 0，才算训练完成。中途不要用另一个训练命令复用相同 `name`；断线先检查进程及日志，不要盲目重启。

## 选权与下一步

```bash
python scripts/select_semifinal_checkpoint_v2.py \
  --run-dir runs/semifinal/grid2x3_ft15_lr3e5_b2 \
  --data data/semifinal_yolo_grid_v1/steel_defect.yaml \
  --baseline best_score_proxy_epoch25.pt \
  --output runs/semifinal/grid2x3_ft15_lr3e5_b2/checkpoint_selection_v2.json \
  --batch 2 --workers 0 --device 0
```

如果平台上还保留 `data/official_yolo_hybrid_v2/steel_defect.yaml`，可额外加入 `--secondary-data data/official_yolo_hybrid_v2/steel_defect.yaml`，让新权重同时通过另一种开发集视图的门槛。

选权报告的 `new_checkpoint_passed` 若为 `false`，不要拿新模型替换旧模型。若为 `true`，仍需在原始 459 张有标注开发图上，用**实际将要提交的单权重推理和后处理流程**比较 AP、召回、框数，不能只看裁块验证指标。复赛 Test 无标签，不能在本地知道官方分数。请把 `checkpoint_selection_v2.json`、训练 `completion.json` 和所选 checkpoint 发回来，再制作正式提交包；不要把不同训练阶段权重简单集成。
