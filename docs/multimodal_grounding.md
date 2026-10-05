# 基于大模型的多模态视觉理解与推理 Baseline

本目录新增的是面向 AIC 赛题的离线可跑 baseline。赛题输入为可见光 RGB、热红外 RGB、16-bit 深度图和英文 Query，输出为可见光图像上的归一化 bbox：`[x1, y1, x2, y2]`。官方成绩使用 `ACC@0.5`，即预测框与真值框 IoU 大于等于 0.5 的查询比例。

当前 baseline 用 7 通道视觉输入（RGB 3 + 红外 3 + 深度 1）和文本哈希编码做跨模态注意力回归。它的目标是先把数据读取、训练、验证、推理和提交 JSON 全链路跑通；正式比赛数据到位后，可以替换视觉/文本编码器为 CLIP、GroundingDINO、Qwen-VL、InternVL 等开源模型。

## 数据审计

```powershell
python scripts/mm_analyze_data.py `
  --data "D:\BaiduNetdiskDownload\基于大模型的多模态视觉理解与推理-示例数据\sample.json" `
  --require-bbox
```

## 样例训练

示例数据只有 1 条标注，训练结果只能用于验证代码能跑通，不能代表泛化能力。

```powershell
python scripts/mm_train.py `
  --data "D:\BaiduNetdiskDownload\基于大模型的多模态视觉理解与推理-示例数据\sample.json" `
  --output runs\multimodal_grounding\sample_debug `
  --epochs 20 `
  --img-size 320 `
  --batch-size 1 `
  --workers 0 `
  --device auto
```

## 推理与提交

推理脚本会保留官方 JSON 中除 `bbox` 外的字段，仅替换 `bbox`，并可额外生成 zip 包。

```powershell
python scripts/mm_infer.py `
  --data "D:\BaiduNetdiskDownload\基于大模型的多模态视觉理解与推理-示例数据\sample.json" `
  --checkpoint runs\multimodal_grounding\sample_debug\best.pt `
  --output runs\multimodal_grounding\submission.json `
  --zip runs\multimodal_grounding\submission.zip
```

## 可视化标注

```powershell
python scripts/mm_visualize_sample.py `
  --data "D:\BaiduNetdiskDownload\基于大模型的多模态视觉理解与推理-示例数据\sample.json" `
  --output runs\multimodal_grounding\sample_bbox.jpg
```

## 后续增强方向

- 用开源视觉语言模型替换哈希文本编码器，优先尝试 CLIP 文本塔 + 多模态视觉塔。
- 对 RGB、红外、深度做独立编码后用跨模态注意力融合，而不是早期通道拼接。
- 加入 bbox heatmap 或 token-to-region 对齐损失，提升小目标定位稳定性。
- 使用外部公开 Visual Grounding 数据预训练，再用官方训练集微调。
- 对深度无效区域、红外弱对比和遮挡样本做定向增强。
