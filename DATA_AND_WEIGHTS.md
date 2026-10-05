# 数据、权重与大文件说明（GitHub 未包含部分）

本仓库保存的是**代码、配置、文档、实验记录和小型模型/产物**。原始工作目录约 **83 GB**，
其中数据集、训练权重和构建输出因体积远超 GitHub 限制（单文件硬上限 100 MB、建议仓库 <1 GB）
未纳入版本库，通过 `.gitignore` 排除。以下说明这些目录的内容、大小与重建方式。

## 被排除的目录

| 目录 | 大小 | 内容 |
|---|---:|---|
| `data/` | ~35.6 GB | 官方训练/开发/测试图像与 XML 标签、增强图像、复赛 Test 788 图、hybrid v2 网格裁剪 |
| `delivery/` | ~20.4 GB | 交付包，含 `iron_4090_full_20260831.zip`（约 20.8 GB） |
| `runs/` | ~19.8 GB | 训练/推理输出、检测器权重（`.pt`）、候选缓存、group-OOF 分析 |
| `submissions/` | ~2.8 GB | 历史 `submission.json`（最大约 908 MB）与提交 zip |
| `fresh_20261004/` | ~2.6 GB | 10-04 的新一轮运行产物 |
| `analysis/` | ~1.7 GB | 归档分析包（含各版本结果副本、权重） |
| `tmp/` | ~0.08 GB | 临时脚本与中间文件 |

此外被忽略：`__pycache__/`、虚拟环境、`.ultralytics/`、`.aris/`、`*.log`、`*.tar.gz`。

## 已纳入仓库的关键内容

- `scripts/`：训练、推理、融合、重排序、校验、实验等全部脚本；
- `steel_defect/`：核心 Python 包（类别、几何、推理、指标、训练器等）；
- `configs/`、`tests/`、`docs/`、`reports/`；
- `refine-logs/`：V13–V34 的完整实验计划、追踪与结果分析；
- `review/`：V12 代码审查与提分方案；
- `updates/`：历次云端推理/训练包（zip，体积较小）；
- `semifinal_v12_perclass_ranker_20261001/`：V12 按类别排序器（含小型 XGBoost `.ubj` 模型）；
- 根目录 `yolo11n/s/m.pt`：官方预训练权重（均 <50 MB）。

## 复现要点

1. 依赖见 `requirements.txt`（Python + Ultralytics + XGBoost + PyTorch）。
2. 数据按 `data/` 的原始目录结构放置（训练/开发/复赛 Test）。
3. 训练与推理流程见各版本 README 与 `refine-logs/`。
4. **类别白名单（重要）**：复赛"板材表面缺陷检测"平台只接受 **8 类**，
   提交中出现 `qilie`（气裂）会整包判失败（`Unknown category_name values: ['qilie']`）。
   尽管通用规则范例中出现过 qilie，实际复赛评测器拒绝该类，务必使用 8 类并以
   `validate_large_submission.py --exclude-category qilie` 校验。
5. 评分口径：`score = 100 × (0.2·Precision + 0.6·Recall + 0.2·AP50)`，
   P/R 为微平均、AP50 按类别宏平均，JSON 中每个框都计入 Precision。
