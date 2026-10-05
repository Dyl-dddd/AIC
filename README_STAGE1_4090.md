# 第一阶段：4090 单权重双视图对照

这是评估包，不启动训练，不生成正式测试集提交，不修改旧权重或数据。所有输出位于本包的 results。先完成对照，再由结果选择下一轮云端训练。

## 1. 上传并独立解压

将 semifinal_stage1_eval_20260923.zip 上传至 /mnt/proj，然后执行：

```bash
cd /mnt/proj
python3 -m zipfile -e semifinal_stage1_eval_20260923.zip /mnt/proj
cd /mnt/proj/semifinal_stage1_20260923
```

压缩包自带 semifinal_stage1_20260923 顶层目录。不要解压覆盖 /mnt/proj/iron。首次安装使用新目录；重复安装前保留已有 results。

## 2. 检查GPU与文件

```bash
nvidia-smi
/mnt/proj/iron/.venv/bin/python scripts/run_semifinal_stage1.py --project-root /mnt/proj/iron --preflight-only
```

先确认 nvidia-smi 没有其他训练任务占用同一GPU。本脚本固定使用可见设备0，要求型号包含4090；不自动切CPU。默认文件：

- /mnt/proj/iron/best_score_proxy_epoch25.pt
- /mnt/proj/iron/runs/semifinal/grid2x3_ft15_lr3e5_b2/weights/epoch5.pt
- /mnt/proj/iron/data/official/train
- /mnt/proj/iron/data/official_v2_20260831b/splits.json

权重和划分均校验SHA256，源图和XML按冻结清单逐个核对。找不到文件时用 --baseline /实际路径/旧权重.pt 和 --candidate /实际路径/epoch5.pt；不要拿别的同名文件替换。不要自动升级torch/ultralytics。GPU尚未能从本地连接，云端环境是否就绪以本步骤结果为准。

## 3. 先跑两张源图的冒烟测试

```bash
set -o pipefail
/mnt/proj/iron/.venv/bin/python -u scripts/run_semifinal_stage1.py --project-root /mnt/proj/iron --limit 2 --output smoke_results 2>&1 | tee smoke.log
```

必须正常结束并生成 smoke_results/SUMMARY.md。这里只是流程测试，不能当正式精度结果。

## 4. 完整对照

```bash
set -o pipefail
/mnt/proj/iron/.venv/bin/python -u scripts/run_semifinal_stage1.py --project-root /mnt/proj/iron 2>&1 | tee stage1.log
```

串行运行2权重×2推理方式×2评估视图，共8个单元。不同时加载两份模型，不做模型集成。每个单元完整跑459个源图；grid_crops为全部派生网格（2754张），不抽样删除背景裁块。与此前1358张抽样裁块的YOLO.val结果不可直接比较，也不能与原图统计相加。局部可见标签沿用训练的0.35可见度和2像素裁剪规则，此视图仅作诊断。

输出分别报告8类mAP、阈值0.05的P/R/FP和全部导出框指标。预计约0.5–2 GPU小时，仅作规划；以冒烟和前10张进度测速修正。脚本按单元保存，若断开后确认原任务已停止，可重跑同一命令，跳过签名匹配且完成的单元；不支持单元内部续跑。切勿在原任务仍运行时启动第二份。

## 5. 返回结果

成功结束后执行：

```bash
cd /mnt/proj/semifinal_stage1_20260923
tar -czf /mnt/proj/semifinal_stage1_results_20260923.tar.gz results stage1.log
```

下载 /mnt/proj/semifinal_stage1_results_20260923.tar.gz 并回传。若失败，先发报错末尾或 stage1.log，不重新训练。候选缓存和结果用于下一步后处理分析；不要删除。

## 已补充官方计分公式

用户提供答疑：总分=20×Precision+60×Recall+20×mAP@0.5，P/R/AP均按0–1输入。历史0.0001/0.9877/0.1044代入为61.352，与61.35一致。

本包用八类micro P/R与宏AP50计算 score_proxy_20_60_20，仅称本地参考分；AP插值、聚合细节仍需核对。原图与裁块分别报告，不把同源视图当独立样本合并。

每个单元会扫描0.001/0.003/0.01/0.03/0.05/0.1/0.2输出阈值，每次都对同一份过滤后的预测重算P/R/AP；不能把高阈值P/R和低阈值AP拼接评分。固定阈值0.05的旧诊断字段只用于横向回归，选方案以threshold_scan里完整输出参考分为主。

验收调整：优先比较完整原图dev参考分及裁块迁移，候选希望比基线提升≥1分并经独立检验。1个百分点Recall下降需要Precision与AP的百分点增益之和超过3才能抵偿；不再把召回下降≤1pp作为不可放宽的硬门槛，也不允许仅因AP提升就放行。不会自动启动训练，本包不保证官方70分。
