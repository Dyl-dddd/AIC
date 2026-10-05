# 复赛重训练执行说明（2026-09-21）

## 已确认的数据变化

- 复赛测试集共 788 张 8-bit 灰度 JPG。
- 488 张为 4096×3000，延续初赛整图形态。
- 300 张为 1387×1516，文件后缀为 `_行_列`，对应 4096×3000 原图的 2×3 重叠裁块。
- 精确裁块坐标为：x=`0,1354,2709`，y=`0,1484`，相邻裁块约重叠 32 px。
- 复赛样本整体比训练/初赛样本更暗、对比度更低；训练对比复赛的 256-bin 灰度直方图 JS 散度为 0.03446，明显高于训练对比初赛的 0.00511。

因此本轮不从头盲训大模型，而从初赛最强的 epoch25 权重出发，使用与复赛完全一致的 2×3 裁块训练视图，加轻量亮度增强做 30 轮微调。

## 本地准备

原始 ZIP 保留不动。已经生成的复赛目录为：

```text
data/semifinal/test
```

如需在 4090 机器重新提取：

```bash
python scripts/extract_test_data.py \
  --test-zip '/mnt/proj/复赛测试集.zip' \
  --output data/semifinal/test \
  --report refine-logs/semifinal_archive_audit.json
```

## 4090 训练命令

将已下载的 `best_score_proxy_epoch25.pt` 放到项目根目录，然后执行：

```bash
cd /mnt/proj/iron
source .venv/bin/activate

python scripts/prepare_data.py \
  --config configs/data/semifinal_grid_v1.yaml

python scripts/train.py \
  --config configs/train/semifinal_finetune_4090.yaml
```

配置采用 `batch=2`、`workers=0`、`cache=false`、AMP，避免共享内存和主机内存溢出。训练不逐轮验证，只在每 5 轮保存 checkpoint。

## 每 5 轮统一验证并选权重

```bash
python scripts/select_semifinal_checkpoint.py \
  --run-dir runs/semifinal/grid2x3_ft30_b2 \
  --data data/semifinal_yolo_grid_v1/steel_defect.yaml \
  --output runs/semifinal/grid2x3_ft30_b2/checkpoint_selection.json \
  --batch 2 --workers 0 --device 0
```

选择规则优先最大化宏召回率，mAP50 仅作为并列时的次级指标。然后用输出 JSON 中的 `best.checkpoint` 覆盖推理命令的权重：

```bash
python scripts/infer.py \
  --config configs/inference/semifinal_grid_v1.yaml \
  --weights /absolute/path/from/checkpoint_selection.json
```

## 分数判断

复赛测试集没有 XML 真值，不能在本地计算真实复赛分数。初赛最强提交为 97.48；如果评分公式、类别和 IoU 规则不变，当前旧权重的合理预期区间约为 94–97，完成本轮匹配式微调后的目标区间为 96–98。该区间是基于尺寸占比和域偏移的工程预测，不是已测成绩。

上线前的硬门槛：新权重必须同时满足裁块 dev 召回率上升、原图 dev 召回率不明显下降；否则继续使用旧 epoch25 权重。

## 合规说明

复赛测试图片只用于无标签分布审计和推理，不把测试预测当作真值回灌训练。若赛事规则明确允许伪标签或半监督训练，再单独做受控实验；默认流程不使用测试集伪标签。
