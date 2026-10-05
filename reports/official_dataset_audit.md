# 2026 AIC 板材缺陷：本轮独立数据审计

审计时间：2026-10-04（北京时间）。脚本：[audit_dataset.py](../fresh_20261004/audit_dataset.py)。本轮只读取当前项目中的官方数据目录，不读取旧实验结果或权重。全部统计来自逐张解码后的 JPG 与逐份解析的 XML。

## 数据定位与范围

- 训练图像及对应 XML：`D:/全球人工智能算法/data/official/train`，各 3200 份，逐一配对，无缺失或孤立 XML。
- 无标签复赛测试图像：`D:/全球人工智能算法/data/semifinal/test`，788 张 JPG；目录另有一个非图像 JSON，未作为测试图像。
- 官方公开[赛题说明](https://www.aicomp.cn/tracks/tracks-6/4174.html)及[复赛通知](https://www.aicomp.cn/notice/notice-3/5248.html)已核对。公开赛题要求 JSON 列表，每条含 `image_id`、`category_name`、绝对像素 `bbox`、`score`；禁止测试伪标签回灌和简单多模型集成。
- 当前项目目录未找到原始 `train.zip`、复赛测试集 ZIP 或评分脚本。因此本报告能证明对上述已提取目录的全量扫描，不能证明这些文件与最初下载 ZIP 逐字节一致。训练目录的 XML 已标准化这一点也须与原始包核对；本轮未修改它们。

## 全量训练统计

3200 张图像全为 4096×3000；含 5889 个框，1074 张空标注图。类别表来自实际 XML：

| 类别 | 含该类图片 | 框数 | 框宽中位数 | 框高中位数 | 细长框比例* |
|---|---:|---:|---:|---:|---:|
| gunyin | 236 | 253 | 103 | 33 | 14.6% |
| huashang | 60 | 153 | 28 | 404 | 88.9% |
| jiaza | 113 | 220 | 24 | 125 | 50.9% |
| jieba | 604 | 1190 | 41 | 74 | 7.4% |
| mamianmakeng | 309 | 2436 | 62 | 53 | 0.1% |
| qilie | 28 | 69 | 41 | 183 | 44.9% |
| yanghuatiepi | 358 | 492 | 40 | 34.5 | 0.8% |
| yiwuyaru | 464 | 603 | 65 | 87 | 10.1% |
| zonglie | 296 | 473 | 46 | 693 | 93.2% |

\*细长定义为长短边之比至少 5。全部目标中，短边小于 8 原始像素的有 3 个，短边小于 16 的有 210 个；细长框 912 个，距任一图像边缘不超过 16 像素的框 530 个。若将 4096 宽整图等比缩到 1280，约 1122/5889 个框的短边将小于 8 输入像素；其中 jiaza 124/220、huashang 71/153、yanghuatiepi 178/492。该尺度计算说明必须检验局部高分辨率视图，尚不证明其能涨分。

文件名提供两个来源代理：数字前缀 1952 张、`C` 前缀 1248 张。其类别组成差异很大：数字来源有 mamianmakeng 2321 框、zonglie 82 框；`C` 来源分别为 115 和 391 框。`RawNN`/`VNN` 仅按文件名视为相机代理，未经官方确认，不解释为真实设备 ID。

XML 与图像尺寸匹配；当前目录内未发现无效或越界坐标，发现 4 个完全重复的类别+坐标框，见 [data_issues.json](data_issues.json)。这不排除语义漏标。空标注图和纹理较强背景需要人工核查训练集后才可作为困难负例。

## 近似重复与冻结划分

用 64 位差异哈希发现 163 个同哈希候选组；同哈希仅是候选，特别是弱纹理图会碰撞。原始组级 80/20 划分的前 8 个候选成员之间有 303 对跨划分同哈希图，仅 1 对同时满足 128×96 预览 MAD<3、相关系数>0.985。该对所在的整个验证组（12 张）已移入训练侧。最终冻结划分：[group_split_frozen.json](group_split_frozen.json)，训练 2538 张、验证 662 张、SHA-256 `74178dd5d8b0509344ce77e5f4f12253e88840e175d129c50814956a0d32b186`。验证 9 类均有标注；最少 jiaza 29 框、qilie 26 框，细分类结论仍有较大抽样不确定性。

该候选检查只保存每个哈希组前 8 个成员，不能证明绝无其他内容近似重复。训练与验证的文件名前缀采集组零交集；后续需要继续使用此冻结清单，不按图像随机重划。

## 产物

- [train_statistics.csv](train_statistics.csv)：逐图尺寸、通道、灰度、局部对比、清晰度与纹理统计。
- [test_statistics.csv](test_statistics.csv)：全部 788 张无标签 Test 的同口径统计。
- [class_distribution.csv](class_distribution.csv)、[bbox_distribution.csv](bbox_distribution.csv)：逐类与逐框统计。
- [train_gt_examples.png](train_gt_examples.png)：仅训练图像与 GT 的抽样可视化。
- [class_distribution.png](class_distribution.png)、[train_test_histogram.png](train_test_histogram.png)、[train_test_features.png](train_test_features.png)：统计图。
- [domain_shift_analysis.md](domain_shift_analysis.md)：按整图和尺寸匹配裁剪视图比较 Train/Test。

这些文件只在 `reports/` 生成；官方图像和 XML 均未被写入。
