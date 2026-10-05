# 全高条带视图（Full-height strip views）：面向长目标定位的训练与推理支持

> 立项依据：`refine-logs/V25_TEST_AWARE_TRAINING_STRATEGY_20261003.md`（规格已写、从未执行）、
> `refine-logs/V41_V12_TEST_AWARE_DIAGNOSIS_20261004.md`（zonglie 定位为最大瓶颈）。
> 目标：为 V12（及后续版本）的候选管线提供"完整纵向范围"的检测信号，提升 zonglie 等
> 长目标类别的 AP50；推理侧改动不需要重训即可使用。

## 1. 问题的证据链

- 官方 V12（67.92）：`P=0.0053`、`R=0.9754`、`AP50=0.4646`。价格杠杆上 AP50 是唯一
  有意义的一项（micro P/R 的杠杆已被 V18/V22/V31/V33/V35 反复证伪；22 个 FN 全找回
  也只有 +1.48）。
- 冻结 dev 上 zonglie 的 IoU 敏感性（V41）：原图视图 AP30 0.6017 → AP50 0.2591 →
  AP75 0.0401——框"差不多对"但到不了 IoU 0.5。
- 定位错误审计（V25）：预测框相对同类真值**高度比中位数 0.351、宽度比 1.362**——
  典型症状是纵向范围不完整。
- 根因：1024 方形切片每次只能看到长目标的一个片段。模型在训练和推理时都只在片段上
  操作，没有一次前向能看到完整的纵裂范围；推理端碎片框与完整框混排时，碎片不命中
  IoU 0.5 而完整框根本不存在。
- 项目历史：V25 规格提出"全高窄条视图（宽 768–1024、高 3000）"但从未执行；V26 以
  COCO 预训练方式训练全高检测器失败（过度过采样纵向条带）；V28 的条带推理依赖了
  失败的 V26 权重。**"从现有强权重直接做条带推理/训练"这一条尚未被尝试过。**

## 2. 设计

- **几何**：`steel_defect.geometry.generate_strip_tiles(width, height, strip_width, overlap)`
  —— 全高、宽度 `strip_width`（默认 810）、沿横向按 `overlap`（默认 0.2）滑动；
  与方形切片共用同一 `_positions` 步进逻辑，首尾边缘对齐（末条贴右边缘）。
  对 4096×3000：7 条（x = 0, 648, 1296, 1944, 2592, 3240, 3286，末条贴右边缘）。
- **推理**：`InferenceOptions.strip_pass`（默认关）在其后追加条带视图；条带候选与方形
  切片候选进入同一 `merge_candidates`（类内 NMS/WBF，按分数竞争）。
  - 条带可独立设置 `strip_imgsz`（0 = 复用主 `imgsz`）；条带与主视图的批次严格分离；
  - 坐标映射与方形切片同构：1:1 像素裁切、仅加 `tile.x` 偏移；内部边缘触碰标记
    （右侧条带右缘、非首条左缘）复用现有 `edge_score_factor` 语义。
- **训练数据**：`prepare_data.py --tile-layout strips|sliding_strips`
  （配 `--strip-width` / `--strip-overlap`）生成条带训练视图；每条条带以原分辨率保存
  完整纵向范围；标注经与方形切片相同的 `clip_box_to_tile` 处理（"跨条带长目标保留
  可见段"的既有策略直接生效）。元数据新增 `is_strip`；`summary.json` 新增
  `strip_views` / `train_strip_views`；`sliding_strips` 与既有 `both`（sliding+grid）
  正交，不影响任何现有布局的行为。
- **缓存与冻结**：`eval.py` 的候选缓存签名**仅在** `--strip-pass` 启用时扩展（追加
  `strip_views`），未启用时签名与旧缓存逐字节一致（旧缓存全部有效）；final holdout
  的冻结 policy 在启用条带时把 `strip_views` 纳入校验。

## 3. 与 V26/V28/V27 的区别

- **不依赖任何新权重**：条带视图由现有主检测器（如 V12 的 Model A/B）直接推理；
- **不改变现有训练配方**：训练侧条带为可选布局，与方形切片并存；
- **原生完整范围**：条带看到的是 1:1 完整纵向像素，不是从片段外推（对比 V27 的两两
  拼接，后者受 "间隔 ≤ 80px / 横向重合" 的保守规则限制）。

## 4. 如何用它提升 V12（操作路径）

1. **候选生成（GPU/云端）**：用 V12 管线的主检测器权重加条带视图导出候选：

   ```bash
   python scripts/infer.py \
     --weights <model_a.pt> --source <test_dir> \
     --cache-only --strip-pass --strip-width 810 --strip-overlap 0.2 \
     --candidate-cache strip_candidates.jsonl.gz
   ```

   对照实验（冻结 dev，不碰 Test）：`scripts/eval.py` 同参数 + `--reuse-candidates`
   复用旧缓存，比较"有/无条带"的同一协议指标。

2. **合并策略**：先做最小版本（条带候选与主候选直接 NMS 合并，不加新排序器）；
   若 dev 对照通过项目既有门槛（双视图 + 分层稳定），再考虑把条带分数作为排序器的
   新支持特征（复用 V12 生成器的"支持框匹配"机制）。

3. **官方探针**：复赛"取最优"使探针零风险（不会覆盖 67.92）；只有官方分数上涨才
   更新保护版本。

## 5. 本地验证（本次交付）

- `tests/test_strip_views.py`（8 项）：
  - 几何：4096×3000 → 7 条全高条带、步进 648、末条贴边；窄图单条；参数校验；
  - 推理：`strip_pass` 坐标偏移只加一次（20 切片 + 7 条带 = 27 视图）；
    `strip_imgsz=1536` 时条带批与方形批严格分离（imgsz 1024 / 1536）；
    关闭时视图数与旧行为逐一致（20）；
  - 数据生成：`strips` 布局在合成 4096×3000 图上端到端产出（含跨条带完整框标注的
    逐项数值校验）。
- **未验证声明**：条带视图对官方分数的增益**尚未验证**；训练侧条带配方需要用真实
  训练数据跑一次 GPU 训练后才能评估（本仓库不含训练数据/权重）。任何数字承诺必须
  来自官方提交或冻结 dev 对照，而不是本文件。

## 6. 变更清单

- `steel_defect/geometry.py`：`generate_strip_tiles`
- `steel_defect/inference.py`：`strip_pass` / `strip_width` / `strip_overlap` /
  `strip_imgsz`；`_flush` 支持按视图设置 imgsz
- `scripts/prepare_data.py`：`strips` / `sliding_strips` 布局；`is_strip` 元数据；
  摘要字段
- `scripts/eval.py` / `scripts/infer.py`：`--strip-pass` 等 CLI 参数；缓存签名 /
  冻结 policy / 缓存头（条件式扩展，向后兼容）
- `tests/test_strip_views.py`：新增测试
- 本文件
