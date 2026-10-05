# 4090 优化训练增量包 v1

本包只新增代码和配置，不包含也不覆盖竞赛数据、旧权重及 `runs/` 历史结果。默认继续使用：

- `data/official/train`
- `data/official_v2_20260831b/splits.json`
- `data/official_yolo_hybrid_v2`
- 现有 `.venv` 与 `yolo11s.pt`

## 设计依据

当前4090五轮原图验证已明显优于本机模型，但尾类指标很低。主线保持已验证更好的 stock YOLO11s，不直接启用此前诊断未通过门槛的P2。新增两组可归因实验：

1. `repeat3`：只改变稀有类别采样，训练视图约增加10%，不复制图像文件。
2. `repeat3_aug`：在repeat3上增加保守的亮度、平移和缩放扰动。

二者都采用 batch=2、workers=0、FP32，规避之前的共享内存错误。训练期间跳过冗余的切片验证，但仍保存每轮权重；训练后基线在第1/3/5轮做完整459张原图验证，长训每5轮做一次原图验证。每组实验同时保留两个选择结果：`best_original.pt` 按常规 mAP50 选择，`best_score_proxy.pt` 按当前公开榜更相关的微平均召回率选择，避免为了冲榜误删常规质量证据。

## 上传和解压

把ZIP和`.sha256`上传到 `/mnt/proj`，先校验，再解压到已有工程：

```bash
cd /mnt/proj
sha256sum -c iron_4090_opt_v1_1_hotfix_code_only.zip.sha256
python3 -m zipfile -e iron_4090_opt_v1_1_hotfix_code_only.zip /mnt/proj/iron
cd /mnt/proj/iron
source .venv/bin/activate
```

ZIP不包含顶层 `iron/`，必须解压到 `/mnt/proj/iron`。它只增加新路径，不覆盖 `cloud.py`、数据或历史实验。

## 准备与检查

以下命令只生成文本采样清单和一个数据YAML，几乎不增加磁盘占用：

```bash
python cloud_opt.py prepare
python cloud_opt.py verify --full
python cloud_opt.py preflight --variant repeat3
```

如果preflight提示GPU不空闲，说明原60轮任务仍在运行；不要同时启动新训练。

## 先跑两个五轮筛选

```bash
python cloud_opt.py run --variant repeat3 --stage baseline --name opt_repeat3_b5 --batch 2 --workers 0
python cloud_opt.py run --variant repeat3_aug --stage baseline --name opt_repeat3_aug_b5 --batch 2 --workers 0
python cloud_opt.py compare
```

`compare` 默认按完整原图的 `recall_micro`、`recall_macro`、`map50_macro` 顺序展示，并同时给出尾类指标；不要看训练日志里的切片占位指标。优先选择召回率更高且 mAP 没有明显崩塌的方案。若增强版更好，进入长训：

```bash
python cloud_opt.py run --variant repeat3_aug --stage long --name opt_repeat3_aug_60 --epochs 60 --batch 2 --workers 0 --after runs/cloud4090_opt/opt_repeat3_aug_b5/original_selection.json
```

若纯重采样更好：

```bash
python cloud_opt.py run --variant repeat3 --stage long --name opt_repeat3_60 --epochs 60 --batch 2 --workers 0 --after runs/cloud4090_opt/opt_repeat3_b5/original_selection.json
```

状态检查：

```bash
python cloud_opt.py status
tail -f runs/cloud4090_opt/_queues/实验名/train.log
```

不要重复使用已经存在的实验名，也不要在旧任务仍占用GPU时并行启动。长训后先比较 `best_score_proxy.pt` 与 `best_original.pt` 的完整原图结果，再决定提交权重。暂时不要使用 `finalize`；模型、推理和集成策略冻结后再使用最终留出集。
