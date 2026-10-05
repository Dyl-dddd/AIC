# 接手说明（HANDOFF）

> 一页纸让接手者快速跑起来、并知道哪些路已经走过/走不通。

## 1. 项目是什么

AIC 产业命题赛（AI+钢铁）复赛赛题 **板材表面缺陷检测**：对钢板表面图像做目标检测，
输出每个缺陷的框、类别、置信度。

- **当前成绩（保护分，V12）**：**67.92**（R=0.9754、P=0.0053、AP50=0.4646，TP=872/FP=164,549/FN=22）
- 复赛 Test：788 图（488 张 4096×3000 整图 + 300 张 1387×1516 网格裁剪）
- 评分公式：`score = 100 × (0.2·Precision + 0.6·Recall + 0.2·AP50)`
  P/R 为微平均、AP50 按类别宏平均；**JSON 中每个框都计入 Precision（没有"工作阈值"）**。

## 2. 仓库地图

- `steel_defect/`：核心 Python 包（类别、几何、推理、指标、训练器、排序器等）
- `scripts/`：训练 / 推理 / 融合 / 重排序 / 校验 / 实验 全部脚本
- `configs/`：配置；`tests/`：单元测试
- `docs/`：赛题分析；`reports/`：数据统计图例
- `refine-logs/`：**V13–V34 完整实验计划、追踪、结果分析（最重要的踩坑记录）**
- `review/`：V12 代码审查与提分方案
- `updates/`：历次云端推理/训练包（zip）
- `semifinal_v12_perclass_ranker_20261001/`：V12 按类别排序器（含小型 XGBoost `.ubj`）
- `DATA_AND_WEIGHTS.md`：被排除的大目录说明
- 根目录 `yolo11n/s/m.pt`：官方预训练权重（<50MB）

## 3. 快速上手

1. Python 环境：`pip install -r requirements.txt`（Python + Ultralytics + XGBoost + PyTorch）。
2. 数据按 `DATA_AND_WEIGHTS.md` 的目录结构放到 `data/`（训练/开发/复赛 Test）。
3. 云端环境（实际跑训练/推理处）：根目录 `/mnt/proj/iron`，
   解释器 `/mnt/proj/iron/.venv/bin/python`，RTX 4090。
4. 各版本运行方式见根目录对应 `README_*.md` 与 `refine-logs/`。

## 4. 必须知道的结论与教训（避免重复踩坑）

- **类别白名单只有 8 类**：jieba、zonglie、jiaza、yiwuyaru、huashang、mamianmakeng、
  yanghuatiepi、gunyin。**提交里出现 `qilie`（气裂）会整包判失败**
  （官方错误：`Unknown category_name values found in submission: ['qilie']`）。
  尽管通用规则范例里出现过 qilie，复赛评测器拒绝它；提交前务必用
  `validate_large_submission.py --exclude-category qilie` 校验。
- **纯后处理/分数重排已被反复证伪**：V14(67.91)、V19(67.92)、V22(67.91)、V31(67.87)、
  V33(67.88) 五次独立分数变换官方非平即降，dev 代理系统性高估约 +0.86。
  FP 裁剪/verifier（V16/V18/V20/V21）、纵裂几何外扩、图级上下文、qilie 补类、
  V37 qilie 专精（dev q AP 0.0077）均无效。
- 量化杠杆：22 个 FN 全找回仅 +1.48；删 50% FP 仅 +0.10；AP50 macro 每 +0.01 = +0.20。
  **真正的提分要靠检测器本身（如 V34 单一检测器适配），不是后处理。**
- 复赛每天 ≤5 次提交、取最优；失败/低分探针不覆盖保护分。

## 5. 需要另行传输的大文件（GitHub 放不下，走网盘）

| 内容 | 目录 | 约大小 |
|---|---|---:|
| 训练/开发/测试数据与标签 | `data/` | 35.6 GB |
| 交付包（含 iron_4090_full zip） | `delivery/` | 20.4 GB |
| 训练输出与检测器权重 `.pt` | `runs/` | 19.8 GB |
| 历史提交 `submission.json`/zip | `submissions/` | 2.8 GB |
| 新一轮运行产物 | `fresh_20261004/` | 2.6 GB |
| 归档分析 | `analysis/` | 1.7 GB |

打包上传网盘前，运行下面脚本生成带 SHA256 的清单，一并交给接手者校验：

```bash
python scripts/make_large_file_manifest.py --dirs data delivery runs submissions fresh_20261004 analysis --out LARGE_FILE_MANIFEST.tsv
```

其中**最关键、务必单独确认**的运行期文件：Model A（imgsz1024）、Model B（imgsz1280）
两个检测器权重，以及云端 TTA 候选缓存
`runs/semifinal/final_submission_tta_v1/test_candidates_tta_1e5.jsonl.gz`、
`runs/semifinal/ensemble_v7_tta_submission/test_model_a_tta_1e5.jsonl.gz`
（有了缓存可不必重跑推理就能继续做后处理实验）。
