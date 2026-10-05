# 真实赛题数据审计与代码适配

审计日期：2026-08-31。审计对象为用户提供的 `train.zip` 与 `初赛测试集.zip`。两个源 ZIP 全程只读，未被解压覆盖或修改；压缩包内只有 JPG/XML 数据，没有需要执行的文档指令。

## 数据结论

| 项目 | 训练集 | 初赛测试集 |
|---|---:|---:|
| 图像 | 3200 | 669 |
| XML | 3200 | 0 |
| 图像/XML 配对缺失 | 0 | — |
| 分辨率 | 全部 4096×3000 | 全部 4096×3000 |
| 实际图像模式 | 全部 8-bit 单通道 `L` | 全部 8-bit 单通道 `L` |
| 标注目标 | 5889 | — |
| 空标注（负样本） | 1074 | — |
| 与另一集合重名 | 0 | 0 |

网页写的是“约 800 张”初赛测试图，当前实际下载包为 669 张；推理必须遍历实际目录，不能硬编码 800。

实际目标计数：

| 类别 | 目标数 |
|---|---:|
| `jieba` | 1190 |
| `zonglie` | 473 |
| `qilie` | 69 |
| `jiaza` | 220 |
| `yiwuyaru` | 603 |
| `huashang` | 153 |
| `mamianmakeng` | 2436 |
| `yanghuatiepi` | 492 |
| `gunyin` | 253 |

最大类与最小类相差约 35.3 倍。框宽中位数 53 px、框高中位数 63 px、框面积中位数约占原图 0.03%；同时有 193 个框的宽或高超过 1024 px，因此只缩放整图会损失小目标，只切片又会损失超长目标的完整范围。

## 发现的数据问题与处理

- 1615 份 XML 写成 `depth=3`，但对应 JPG 实际为单通道；规范化副本会改成 1。
- 2666 个 `<filename>` 只需要扩展名/格式归一化，65 个文件名 stem 与实际条目不一致；配对永远以 ZIP 中同名 JPG/XML 为准，规范化副本写回真实图像名。
- 1 个框的 `ymax=3001`，超出 3000 高度 1 px；规范化副本裁剪到 3000，并在报告中计数。
- 1074 份空 XML 是有效负样本，保留而不是删除。
- ZIP 中没有路径穿越条目、加密文件、重复 stem；训练和测试图像均可被 Pillow 识别。

## 算法改动依据

1. 新增 `scripts/extract_official_data.py`：只按文件类型和 stem 读取 ZIP，不信任压缩包路径；原始 ZIP 只读；输出使用暂存目录，成功后再原子改名。
2. 修复跨切片长框：当缺陷横跨整个 tile 时保留该 tile 内的有效片段，避免把可见缺陷当负背景。真实标注上比旧规则多保留 120 个长目标片段。
3. 推荐混合视图：1024 重叠切片负责小目标，1024 全局缩放视图负责长裂纹/全局上下文。按当前参数，重复采样前约生成 6613 个正切片、3664 个负切片和 3200 个全局视图。
4. `qilie` 长尾明显，混合配置使用上限 3 倍的稀有类重复和 Focal Loss；它是推荐起点，不替代后续消融验证。
5. 去模糊和 CLAHE 默认关闭。真实背景包含水渍、油污和纹理，未经验证的锐化容易增加误报。

## 训练平台命令

假设两个 ZIP 上传到 `/workspace/input`：

```bash
python scripts/extract_official_data.py \
  --train-zip /workspace/input/train.zip \
  --test-zip '/workspace/input/初赛测试集.zip' \
  --output /workspace/data/official

python scripts/prepare_data.py --config configs/data/official_hybrid.yaml
python scripts/train.py --config configs/train/official_hybrid_24gb.yaml

python scripts/eval.py --config configs/evaluation/official_hybrid_calibration.yaml
python scripts/eval.py --config configs/evaluation/official_hybrid_val.yaml \
  --thresholds runs/evaluation/official_hybrid_calibration/thresholds.json

python scripts/infer.py --config configs/inference/official_hybrid.yaml \
  --thresholds runs/evaluation/official_hybrid_calibration/thresholds.json
```

如果平台中的 ZIP 名称不同，只改第一条命令。推荐至少预留 60 GB 工作空间（若输入 ZIP 使用独立只读挂载可少一些）；混合训练配置已关闭 Ultralytics 的 `.npy` 磁盘缓存，避免再占用数十 GB。
