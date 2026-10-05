# V15 官方逐类别归因探针

V14 官方结果为 `67.91 / mAP50 0.4641`，低于 V12 的 `67.92 / 0.4646`；两版的 Recall、Precision、TP、FP、FN 完全相同。因此当前问题只是 `jieba`、`jiaza`、`mamianmakeng` 三个类别的排序迁移，并非框数量或定位变化。

本包不训练、不推理、不需要 GPU。它读取云端已经生成的 V12 与 V14 `submission.json`，验证两个文件的候选成员、顺序和坐标完全一致，然后在数十秒内生成三个单类别探针和两个保守插值版本。

## 运行

```bash
cd /mnt/proj/iron
unzip -qo semifinal_v15_class_attribution_20261002.zip
cd semifinal_v15_class_attribution_20261002

python -u generate_v15_class_probes.py --project-root /mnt/proj/iron
```

不需要安装任何第三方依赖。

## 第一轮提交顺序

按开发集加权边际增益从高到低，依次提交下面三个文件。每个 ZIP 内严格只有一个 `submission.json`：

```text
/mnt/proj/iron/runs/semifinal/v15_class_attribution_submission/semifinal_v15_probe_mamianmakeng_SUBMIT_ONLY.zip
/mnt/proj/iron/runs/semifinal/v15_class_attribution_submission/semifinal_v15_probe_jiaza_SUBMIT_ONLY.zip
/mnt/proj/iron/runs/semifinal/v15_class_attribution_submission/semifinal_v15_probe_jieba_SUBMIT_ONLY.zip
```

每提交一个就记录官方 `score` 和 `mAP50`。不要把代码包或输出目录整体上传。

## 备用版本

如果当天只剩一个槽位且无法做完归因，优先试更保守的：

```text
/mnt/proj/iron/runs/semifinal/v15_class_attribution_submission/semifinal_v15_blend25_all3_SUBMIT_ONLY.zip
```

`blend50_all3` 的改动强度介于 `blend25_all3` 与 V14 之间，优先级更低。V12 的 67.92 仍是保护基线；在单类别探针没有超过它前，不替换最终最佳提交。
