# 训练平台执行手册

## 当前结论

代码已经形成可运行的单模型 P2 检测流水线，但目前没有真实赛题训练结果，因此不能声称 Focal、去模糊、全局视图或 WBF 已提升精度。默认值是无混杂的中性基线；候选组件必须通过原图 VOC 标注评估后才能保留。

所有路径型 YAML 都是模板，上传平台后先替换 `/workspace/data/...`。

## 1. GC10 预训练

```bash
python scripts/prepare_data.py --config configs/data/gc10_clean.yaml
python scripts/train.py --config configs/train/gc10_pretrain_24gb.yaml
```

GC10 使用 `classes: auto`，不会错误套用赛题的 9 类顺序。迁移到赛题时，10 类检测头会被跳过，只复用形状匹配的骨干和颈部权重。

### GC10 时间怎么估

时间取决于切图后样本数，而不是原图数。先看：

```bash
cat /workspace/data/gc10_yolo_clean/metadata/summary.json
```

在最终 GPU、分辨率和 batch 上先跑 5 epoch：

```bash
python scripts/train.py \
  --config configs/train/gc10_pretrain_24gb.yaml \
  --epochs 5 --name gc10_pilot

python scripts/estimate_runtime.py \
  --results runs/gc10/gc10_pilot/results.csv \
  --total-epochs 80
```

仅用于租卡排期的粗略范围：单卡 24 GB、768 输入、80 epoch 通常约 2–8 小时；T4 或大量切片可能到 8–15 小时。以 5-epoch 实测外推为准，预留 15% 余量。

## 2. 赛题数据生成

先把上传的官方 ZIP 规范化为稳定的 `train/` 与 `test/` 目录（源 ZIP 不会被修改）：

```bash
python scripts/extract_official_data.py \
  --train-zip /workspace/input/train.zip \
  --test-zip '/workspace/input/初赛测试集.zip' \
  --output /workspace/data/official
```

真实包审计确认训练图 3200 张、测试图 669 张、空 XML 负样本 1074 张，并存在 193 个宽或高超过 1024 px 的长框。完整统计见 `docs/official_data_audit.md`。

先生成 75% train、10% calibration、15% untouched val 的分组划分：

```bash
python scripts/prepare_data.py --config configs/data/official_clean.yaml
```

面向最终候选的混合视图配置同时生成 tile 与全局视图：

```bash
python scripts/prepare_data.py --config configs/data/official_hybrid.yaml
python scripts/train.py --config configs/train/official_hybrid_24gb.yaml
```

干净基线明确是：P2、1024 tile、0.25 overlap、BCE、repeat=1、无去模糊、无 CLAHE、无全局训练视图。另生成只改变重复采样的数据：

```bash
python scripts/prepare_data.py --config configs/data/official_repeat3.yaml
```

架构 × 视图的 2×2 还需要只含整图缩放视图的数据：

```bash
python scripts/prepare_data.py --config configs/data/official_full_clean.yaml
```

`official_candidate.yaml` 同时开启多个候选组件，只用于最终候选复核，不能拿它与 clean 的差异归因给某一个算法。

## 3. 训练顺序

先运行 stock/P2 × full/tiled 的 2×2。full 与 tiled 的样本数不同，先用工具算出 matched epochs，并把数值填入 `architecture_view.yaml` 对应 cell，保证优化器更新数一致：

```bash
python scripts/match_update_epochs.py \
  --reference-summary /workspace/data/official_yolo_clean/metadata/summary.json \
  --candidate-summary /workspace/data/official_yolo_full_clean/metadata/summary.json \
  --reference-epochs 100 --reference-batch 6 --candidate-batch 6

python scripts/run_experiments.py --matrix configs/experiments/architecture_view.yaml
```

选定架构/视图后运行 BCE 基线：

```bash
python scripts/train.py --config configs/train/official_baseline_24gb.yaml
```

然后执行 loss × repeat 的 2×2 消融：

```bash
python scripts/run_experiments.py --matrix configs/experiments/core_ablation.yaml
```

中断后重跑同一命令会跳过已有 `best.pt` 的实验；单次训练中断可用：

```bash
python scripts/train.py --config configs/train/official_baseline_24gb.yaml --resume auto
```

显存不足时先把 batch 从 6 调到 4/2，再降低 imgsz；不要一开始就改变 tile、模型和增强，否则实验不可比较。多实验必须保持相同 split、预训练权重、batch、epoch/优化器更新数和 checkpoint 选择规则。

## 4. 原图级评估与阈值校准

Ultralytics 在切片目录上的 val 只用于训练监控，正式结论必须来自原图和原始 XML：

```bash
# calibration 上搜索每类阈值
python scripts/eval.py --config configs/evaluation/official_calibration.yaml

# untouched val 上报告最终结果，不调阈值
python scripts/eval.py --config configs/evaluation/official_clean.yaml \
  --thresholds runs/evaluation/official_calibration/thresholds.json
```

当前 `eval.py` 的每类阈值通过 `--tune-thresholds` 生成。最终 val 不应再次搜索阈值。报告包含 AP50、AP75、AP50:95、宏 P/R/F2、尾类指标和每图误报数，并记录权重与 split 的 SHA-256。

每次评估还会保存 `raw_candidates.json.gz`。可在不重跑网络的情况下公平比较后处理：

```bash
python scripts/eval.py --config configs/evaluation/official_clean.yaml \
  --candidate-cache runs/evaluation/official_clean/raw_candidates.json.gz \
  --reuse-candidates --merge wbf --output runs/evaluation/replay_wbf
```

缓存签名会核对权重、split、tile、预处理和网络推理参数；不匹配会拒绝回放。

## 5. 提交

把 calibration 得到的阈值用于测试集：

```bash
python scripts/infer.py --config configs/inference/official_clean.yaml \
  --thresholds runs/evaluation/official_calibration/thresholds.json
```

提交前确认图片基名无重复、类别顺序与权重完全一致、框不越界且 JSON 可解析。去模糊、全局 pass、WBF、边缘惩罚均保持关闭，除非对应的独立消融在 untouched val 上稳定胜出。

## 6. 最小实验排期

优先顺序：

1. 原图 evaluator 与缓存回放一致性；
2. stock YOLO11s × P2 和整图 × 切图的 2×2；
3. BCE × Focal 与 repeat 1 × 3 的 2×2；
4. 缓存上比较 NMS/WBF、融合 IoU和边缘惩罚；
5. overlap 的训练/推理 2×2；
6. 全局 train/infer 2×2；
7. 去模糊 × CLAHE 2×2；
8. 基线和最终候选各跑 3 个 seed。

若一次 1024 基线训练耗时为 `H`，优先级 1 的训练筛选约为 `9H`；基线和最终候选三种子确认后约为 `13H`。后处理比较复用候选缓存，不计入重训次数。
