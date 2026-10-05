# 待实施代码修改清单

日期：2026-08-31。以下是计划任务，不代表代码已经修改；历史数据、训练日志和权重不得覆盖。  
优先级P0完成后才允许长训。所有正式结果记录代码/标签/划分/依赖版本。

| 优先级 | 位置 | 修改内容 | 必须新增的验收 |
|---|---|---|---|
| P0 | scripts/extract_official_data.py；新私有审计工具 | 全图SHA来源清单、重复组及标注冲突记录；train/test重叠训练侧隔离；保留原ZIP | 同图异名、同图异标注、跨test内容、损坏包、只读与恢复测试 |
| P0 | steel_defect/voc.py；scripts/prepare_data.py | 显式来源group字段/映射；Raw、候选C和重复SHA联合分组；先划分再切片；单独v2输出 | 关联链连通分量、跨视图重复、候选C保守分组；训练/dev/final完全组隔离 |
| P0 | VOC规范化与标注审计 | 精确重复框处理规则；冲突标注隔离/复核；不把所有空XML都删掉 | 训练缓存与原图GT同一标签版本；无静默并集、无误删负样本 |
| P0 | steel_defect/metrics.py；scripts/eval.py | gold fixtures；完整范围强校验；final禁止tune；评估无GT类别FP也计入全局FP/图 | 完美预测AP、重复预测FP、错类、空GT、未知图ID、缺文件与非有限分数 |
| P0 | scripts/eval.py；steel_defect/inference.py | 候选缓存纳入预处理/标签/代码版本；记录每图与每视图来源；分离AP候选门槛与工作阈值 | 代码变更后旧缓存拒用；全局/切片坐标反投影；阈值不伪造AP改善 |
| P0 | scripts/train.py；steel_defect/runtime.py | 整个构模/加载/训练都在失败记录保护内；记录真实optimizer更新、正样本/质量分数、有效batch与LR | 构模失败状态不是starting；加载失败可追踪；中断可恢复且数据/配置hash相符 |
| P0 | scripts/match_update_epochs.py | 更名区分microbatch与optimizer update，计入累积和warmup；优先实测步数 | batch1/2/4+nbs64的等价测试；改变repeat/视图数不冒充等训练量 |
| P0 | scripts/run_experiments.py | best.pt存在不等于完成；核验完成清单、计划epoch、退出码与config/data hash；dry-run不改运行状态 | 部分epoch已有best不得跳过；旧配置同名不得误复用；dry-run只读 |
| P1 | 小样本诊断数据入口与configs | 从v2 train选32–64个清晰正片+8负片，关闭随机增强/过采样；训练=评价仅用于诊断 | 明确diagnostic标记；不把此高分填原图泛化主表；各类样本和种子固定 |
| P1 | stock/P2构模和预训练加载 | stock真正使用公共预训练；P2记录按模块/参数量迁移；语义映射只有测试后启用 | 层语义/形状/参数hash对齐，随机新头单列；不把模型自复制算COCO覆盖 |
| P1 | steel_defect/focal_trainer.py | 保留BCE开关；记录软目标下梯度与目标分数，不先声称alpha是9类平衡权重 | 硬/软目标、全负batch、极小质量分数、AMP有限梯度、小样本对照 |
| P1 | 训练与验证配置 | BCE+repeat1+弱增强基线；原图dev选best_original；final封存 | 同一划分、相同初始化来源、变化因子唯一；原图评估不丢失样本 |
| P2 | 数据采样/增强 | 按错误分析选择repeat、难负样本、亮度/轻模糊等，变换与标注同步 | train-only变换；长目标/微小目标存活率；负片不得含未标可见片段 |
| P2 | 后处理/部署 | 单权重NMS及长框片段对照、阈值冻结；完整669图处理清单 | 相邻同类不误合并、框坐标合法、空预测正常、提交字段严格、无人工补结果 |
| P3 | 架构扩展 | 仅当误差分析和预算支持再考虑新增模块或替代架构 | 必须打败已学习的简单基线，测吞吐/显存/原图精度，无收益移除 |

## 保护规则

- v1结果保留；修复分组后的v2不沿用旧best/last进行有效泛化实验。
- 不能从best.pt文件存在推出任务完成；最终被剥离optimizer的权重只用于推理/新微调，不伪称精确续训。
- 所有标签复核仅限允许用于训练的数据；测试相同图仍只能由模型预测。
- 新生成的大目录做磁盘预算，不删除原始ZIP或用户其他实验来腾空间。
- 当前Ultralytics/PyTorch版本先固定，升级另立独立实验。

