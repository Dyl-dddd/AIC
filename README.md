# 2026 AIC 钢铁板材表面缺陷检测

这是一个面向赛题的端到端单模型基线：从 PASCAL VOC 标注检查、超高分辨率切片、训练，到滑窗推理和官方 `submission.json` 生成。

赛题逐项拆解见 [`docs/competition_analysis.md`](docs/competition_analysis.md)，训练平台命令、GC10 时间估算和无泄漏实验顺序见 [`docs/platform_training.md`](docs/platform_training.md)。

## 方案取舍

- **P2 小目标检测头**：输出步长 4/8/16/32 的四级特征，尽量保留几十像素缺陷。
- **多尺度与全局上下文**：YOLO11 的 PAN-FPN/C3k2 负责多尺度融合，骨干末端 C2PSA 提供全局上下文。
- **长尾优化**：可独立开启稀有类重复与 Focal Loss；默认 BCE + repeat=1，避免基线混杂。
- **高分辨率切片**：训练和推理均使用重叠滑窗，预测框会映射回 4096×3000 原图坐标。
- **单模型提交**：不做赛事禁止的多模型投票或权重平均。
- **去模糊自动门控**：采用清晰度门控的轻量锐化，而不是对所有图片盲目反卷积。功能默认关闭，只在原图级 A/B 证明有效后启用。

> 赛题网页描述的是钢铁板材表面工业相机图像，不是 X 射线焊缝图像。两者都受益于小目标、多尺度和长尾优化，但成像噪声与缺陷语义不同，本工程以赛题网页为准。

## 目录

```text
configs/
  yolo11s-p2.yaml            # 四尺度单模型
scripts/
  extract_official_data.py   # 安全审计并规范化官方训练/测试 ZIP
  analyze_dataset.py         # 数据审计
  prepare_data.py            # VOC -> 重叠切片 YOLO 数据集
  train.py                   # BCE/Focal 配置化训练、恢复与运行清单
  eval.py                    # 原图 VOC GT 评估与候选缓存
  infer.py                   # 滑窗推理 -> submission.json
steel_defect/
  classes.py
  deblur.py
  focal_trainer.py
  geometry.py
  voc.py
tests/
```

## 环境

推荐 Python 3.10–3.12、PyTorch 2.x、CUDA GPU（16 GB 以上更合适）。

```powershell
pip install -r requirements.txt
```

本机当前已验证：Python 3.12、PyTorch 2.5.1、OpenCV 4.11、Ultralytics 8.3.169。

## 1. 数据审计

真实赛题 ZIP 的审计结果与已确认修复见 [`docs/official_data_audit.md`](docs/official_data_audit.md)。先将压缩包安全规范化：

```powershell
python scripts/extract_official_data.py `
  --train-zip D:\BaiduNetdiskDownload\train.zip `
  --test-zip D:\BaiduNetdiskDownload\初赛测试集.zip `
  --output D:\data\official
```

该脚本不会修改源 ZIP；会保留空 XML 负样本、校正 XML 中的通道/文件名，并把唯一的 1 px 越界框裁到图像边界。只检查不解压时加 `--verify-only` 并省略 `--output`。

假设训练集中的图片和同名 XML 位于同一个目录或其子目录：

```powershell
python scripts/analyze_dataset.py --source D:\data\steel\train
```

输出会列出图片数、标注数、类别分布、框尺寸分布、未知类别、缺失配对和清晰度分布。先修复未知类别和损坏标注，再切片。

## 2. 生成切片数据集

```powershell
python scripts/prepare_data.py `
  --source D:\data\steel\train `
  --output D:\data\steel_yolo `
  --tile-size 1024 `
  --overlap 0.25 `
  --val-ratio 0.20 `
  --negative-ratio 0.25 `
  --rare-repeat-cap 1 `
  --no-deblur `
  --no-global-context
```

重要参数：

- `--tile-size`：建议从 1024 开始；24 GB 显存可尝试 1280。
- `--overlap`：建议 0.20–0.30，降低长裂纹在切片边缘被截断的风险。
- `--negative-ratio`：每张原图最多保留的负切片数 / 正切片数；不要完全丢弃负样本。
- `--rare-repeat-cap`：按类别出现频率的平方根反比重复稀有类训练切片；默认 1（关闭），消融时测试 2/3。
- 去模糊默认关闭；传入 `--deblur` 才启用清晰度门控锐化。
- 全局训练视图默认关闭；用 `--view-mode global` 生成整图缩放数据，用 `--view-mode both` 同时加入 tile 与全局视图。
- 推荐增加 `--calibration-ratio 0.10 --val-ratio 0.15`，分别用于阈值校准和最终未触碰验证。
- `--clahe`：可选局部对比度增强。与去模糊一样，必须用验证集确认收益。

划分按文件名前的钢卷/采集组 ID 进行，同组帧不会同时进入训练集和验证集，以减少相邻帧泄漏。

## 3. 训练

```powershell
python scripts/train.py --config configs\train\official_baseline_24gb.yaml
```

针对本次真实数据（小框占多数但同时存在超长框），推荐先跑切片+全局视图混合配置：

```powershell
python scripts\prepare_data.py --config configs\data\official_hybrid.yaml
python scripts\train.py --config configs\train\official_hybrid_24gb.yaml
```

默认训练路径使用单模型 P2 检测头与 BCE。Focal 是显式候选项。首次使用 `yolo11s.pt` 时 Ultralytics 可能下载公开预训练权重；离线环境可传本地权重路径，或用 `--pretrained none` 从头训练。

建议实验顺序：

完整的无混杂消融顺序与计算预算见 `docs/platform_training.md`。任何精度结论都必须来自 `scripts/eval.py` 的原图/XML 评估，不使用切片 val 指标代替。

## 4. 推理并生成提交文件

```powershell
python scripts/infer.py `
  --weights runs\steel\p2_focal\weights\best.pt `
  --source D:\data\steel\test `
  --output submission.json `
  --tile-size 1024 `
  --overlap 0.25 `
  --conf 0.01 `
  --iou 0.55 `
  --batch 8 `
  --device 0 `
  --no-deblur `
  --no-global-pass
```

输出字段严格为：

```json
[
  {
    "image_id": "example.jpg",
    "category_name": "zonglie",
    "bbox": [100, 200, 160, 800],
    "score": 0.91
  }
]
```

坐标是原图绝对像素 `[xmin, ymin, xmax, ymax]`。推理脚本会进行类别内 NMS、坐标裁剪和格式校验。

## 去模糊说明

用户给出的 2018 年文章主要解释卷积/反卷积思路，并明确表示其“聚焦算法”未开源；文章链接的传统盲去卷积需要估计模糊核，计算量大，而且可能把钢板纹理或水渍锐化成伪缺陷。

因此本工程没有复刻一个不存在的完整算法，而采用更稳健的策略：

1. 用拉普拉斯方差估计清晰度；
2. 仅在低清晰度图像上启用；
3. 使用受限的多尺度反锐化掩模；
4. 用亮/暗光晕惩罚限制振铃；
5. 当前默认关闭；可通过 `--deblur` 生成候选数据并在固定原图验证集上比较。

对于竞赛，通常“用模糊增强训练检测器，使模型对模糊鲁棒”比“对全部测试图先强行去模糊”更安全。

## 类别顺序

```text
0 jieba
1 zonglie
2 qilie
3 jiaza
4 yiwuyaru
5 huashang
6 mamianmakeng
7 yanghuatiepi
8 gunyin
```

如果赛方下载包中的真实 XML 使用了不同拼写，以 XML 为准，并同步修改 `steel_defect/classes.py`。不要仅凭网页示例猜测标签。
