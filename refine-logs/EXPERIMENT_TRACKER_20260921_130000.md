# 复赛实验跟踪表

| Run ID | Milestone | Purpose | Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| SF001 | M0 | ZIP 安全审计 | semifinal archive | 复赛测试 | 文件数/尺寸/模式 | MUST | DONE | 788 JPG，0 异常路径 |
| SF002 | M0 | 域偏移分析 | train/preliminary/semifinal | 有界样本 | 灰度、对比度、清晰度、JS | MUST | DONE | train-vs-semi JS=0.03446 |
| SF003 | M1 | 生成匹配数据 | grid2x3 + global + repeat3 | frozen train/dev | 标签/数量审计 | MUST | TODO | 预计约 2 万基础视图，另有稀有类重复 |
| SF004 | M2 | 复赛微调 | epoch25 -> grid2x3_ft30 | frozen train | loss/有限性 | MUST | TODO | batch2, workers0, cache off |
| SF005 | M3 | checkpoint 选择 | epochs 5/10/15/20/25/30 | crop-style dev | recall, mAP50 | MUST | TODO | 统一在训练后验证 |
| SF006 | M3 | 原图回归 | selected vs epoch25 | original dev | micro/macro recall | MUST | TODO | 下降 >0.5pp 则拒绝新权重 |
| SF007 | M4 | 复赛推理 | selected checkpoint | 788 test images | 格式/覆盖率 | MUST | TODO | conf=0.001, max_det=2000 |
| SF008 | M4 | 旧权重回退对照 | epoch25 | 788 test images | 官方分数 | NICE | TODO | 仅在允许提交次数充足时 |
