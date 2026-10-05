# 独立代码审查报告（2026-10-05）

> 审查者：SpikeBot 000（CodeWhale-LOCAL，独立于项目作者）
> 审查对象：`AIC` 仓库 HEAD `67719d0`（初始提交全量代码，490 个 Python 文件）
> 审查方法：全量静态检查（compileall / ruff / pytest）＋ 官方复赛测试集（788 图）真实数据验证 ＋ 核心模块定向研读
> 本文档随修复分支一并提交。修复的 4 处问题见第三节；建议项见第四节。

---

## 一、结论摘要

- **整体评价**：核心库（`steel_defect/`）与提交生成管线的工程质量高：边界条件处理完整、失败模式普遍 fail-closed、训练/评估有审计与冻结契约、产物有 SHA256 校验。审查的重点缺陷集中在**可测试性与仓库卫生**，未发现会导致结果错误的核心逻辑缺陷。
- **本次修复 4 处**（3 类）：核心模块惰性导入（重要）、`prepare_data.py` 线程回调绑定（低风险）、`.gitignore` 去重（卫生）。
- **独立真实验证**：以官方发布的 788 张复赛测试图，验证了 `generate_grid_tiles` 的网格几何与官方裁剪 **100% 一致**（15/15 样本位置精确、亚像素搜索零偏移）；`prepare_data.py` 两条切片管线端到端跑通、逐框数值手工核对精确。

## 二、独立验证证据（正面结论）

### 2.1 网格几何 vs 官方发布裁剪（真实数据）

复赛发布包含 488 张 4096×3000 整图与 300 张 1387×1516 网格裁剪，其中 66 张裁剪存在同名整图。

- 用 `steel_defect.geometry.generate_grid_tiles(4096, 3000, 1387, 1516, rows=2, cols=3)` 生成 6 个 tile，与每张官方裁剪做全图比对：**15/15 样本的最佳匹配 tile 索引都与文件名 `_r_c` 后缀所指位置完全一致**（`idx = r*3 + c`）。
- 对 3 张样本做 ±3 像素的平移搜索：**(0,0) 即全局最优偏移**，无系统性错位。
- 像素并非 bit-exact（mean abs diff 约 2–9 灰度级），这符合"发布整图 JPEG 与裁剪图 JPEG 为两次独立有损编码"的预期；几何层面完全一致。
- 结论：`semifinal_grid` 布局的 `x=[0,1354,2709]`、`y=[0,1484]` 与官方构造精确吻合，该结论支撑了 V12 前后各版本对 grid 布局的假设。

### 2.2 `prepare_data.py` 端到端（真实图像）

以 2 张真实测试图（4096×3000、1387×1516）＋ 手写 VOC XML（含中英文类别别名"划伤/辊印/氧化铁皮"）运行两条管线：

- `sliding` 布局：`rc=0`；输出 3+3 个 tile。逐框手工核对归一化坐标（含跨 tile 裁剪案例：局部框 `x 732..1024` → `cx=0.857421875` 等）**全部精确**；"可见碎片被拒绝时不作为负样本"的逻辑按设计生效（4 个 tile 中 1 个被正确跳过）。
- `semifinal_grid` 布局：`rc=0`；4096 图产出 `x=1354` 列 2 个 tile、1387 图产出整图 1 个 tile，命名带 `__grid2x3__` 标记，符合预期。
- 负采样比例（`negative_ratio=0.25` → 每图至少 1 个负样本）、类别别名映射（中文 → 官方英文类名）、`steel_defect.yaml` 输出结构均正确。

### 2.3 测试套件状况（修复前后）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| pytest 通过 | 37 | **45** |
| pytest 失败 | 10（全部因缺 torch 环境） | 6（参见说明） |
| 收集错误 | 21 | 18 |

- 修复后仍失败的 6 项全部在 `tests/test_governance.py`，其中 3 项本身需要真实 `torch`（构造 optimizer/BatchNorm/tensor），在无 torch 环境失败是正确行为；另 3 项依赖 `scripts/train.py` 的导入链（见 4.1 建议）。
- 收集错误（18 项）全部源于测试模块顶层导入链需要 `torch`/`ultralytics`/`xgboost`；其中 3 项由本次修复解锁（`test_ensemble_v6`、`test_final_submission_cache`、`test_varifocal_submission_rescore`），其余属于"训练/推理集成测试"，需完整 GPU 环境。

## 三、已修复问题（本分支）

### 3.1【重要·可测试性】`steel_defect/runtime.py`、`steel_defect/training_audit.py` 模块级重依赖

- **问题**：两个模块在顶层 `import torch` / `import ultralytics`，导致其中**不依赖任何 ML 框架的纯函数**（`sha256_file`、`load_yaml_section`、`apply_profile_defaults`、`OptimizerAudit`、`freeze_bn_statistics` 等）在无 GPU/无 torch 的环境下不可用，并连带使 13 个以上测试模块无法收集、`prepare_data.py` 等脚本无法在轻量环境加载。
- **修复**：将 `torch`/`ultralytics` 移到实际使用它们的函数内部（`platform_manifest`、`cuda_witness`、`OptimizerAudit.batch`、`audit_initial_transfer`、`save_raw_diagnostic_checkpoint`、`freeze_bn_statistics`）。**行为完全不变**，仅 import 时机变化。
- **效果**：`import steel_defect.runtime` / `training_audit` 在无 torch 环境成功；pytest 通过数 37 → 45；`prepare_data.py` 得以在轻量环境完成 2.2 节的端到端验证。训练热路径（每 batch）增加的函数内 import 为 `sys.modules` 命中，开销可忽略。

### 3.2【低风险·健壮性】`scripts/prepare_data.py:507` lambda 未绑定循环变量（ruff B023）

- **问题**：`executor.map(lambda record: process_record(record, split, ...), ...)` 中 `split` 为闭包引用；当前实现因"立即提交 + 同步消费"而实际安全，但模式脆弱（一旦消费时机改变或异常提前退出，线程可能读到下一轮 split 值）。
- **修复**：`lambda record, split=split: ...` 在定义时绑定。行为不变，消除隐患。

### 3.3【卫生】`.gitignore` 重复条目

- **问题**：文件尾部有 30 行重复条目（`data/`、`runs/`、`tmp/`、`fresh_20261004/`、`analysis/`、`delivery/` 等各出现 2–3 次）。
- **修复**：去重整理。**已用集合等价性验证**：33 个唯一条目 → 33 个，无缺失、无新增；抽查 `git check-ignore` 全部生效。

## 四、建议项（未修改，留待作者决策）

### 4.1 `scripts/train.py` 的 `DEFAULTS` 导入链

`scripts/run_experiments.py` 顶层 `from scripts.train import DEFAULTS`，而 `train.py` 顶层导入 `torch`/`ultralytics`。这使 `test_queue_dry_run_is_read_only` 等 3 个纯逻辑测试必须要有完整 ML 环境。若希望 dry-run 逻辑可独立测试，可将 `DEFAULTS` 拆到独立模块（如 `scripts/train_defaults.py`）或惰性化 `train.py` 顶层导入。**未修改原因**：`train.py` 本质是训练入口、其依赖属固有；拆分属于代码组织决策，尊重作者结构。

### 4.2 V12 排序器生成器的特征列序校验（承接前次 review）

前次审查（`review/V12_CODE_REVIEW_AND_SCORE_PLAN_20261003.md` 问题 #1/#2）提出的"numpy 位置传参、列序漂移会静默出错"与"缓存缺失静默降级"仍然成立。本次独立复核了 `v12_perclass_spec.json`：24 个特征名、8 个类别、`strict_group_oof` 齐备，生成器已校验模型 SHA256 与文件存在性，但特征列仍以 numpy 位置传入 XGBoost。**未修改原因**：该目录是冻结的上线提交包（含逐类模型与部署 SHA256），改动会破坏其可复现性；建议下次重跑/迭代该排序器时落实前次建议（`pandas.DataFrame(matrix, columns=feature_names)` + `booster.feature_names` 断言）。

### 4.3 `updates/` 目录与仓库体积

`updates/` 含 445 个历史交付副本（292 个 .py），与顶层活跃代码大量重复，且 ruff 报告在其中产生约一半的重复条目。**未修改原因**：历史交付留档是作者决策；若希望精简，可考虑迁出为网盘归档或在 README 中注明其"冻结快照"属性。

### 4.4 `focal_trainer.py` 与 `quality_trainer.py` 的 checkpoint 可移植性处理不一致

`quality_trainer.py` 的文档字符串明确说明其设计目标之一是"保存的 `.pt` 不 pickle 项目特定模型子类"（通过仅在训练期注入 criterion、保持 stock `DetectionModel`）；而 `focal_trainer.py` 直接以自定义子类 `FocalDetectionModel` 建模并训练，按 Ultralytics 的保存路径，其 `.pt` 会引用 `steel_defect.focal_trainer.FocalDetectionModel`，跨环境加载需要该包可导入。**未修改原因**：`focal` 路线非当前主线（默认 `bce`），且没有跨环境加载 `.pt` 的确证需求；建议两者对齐或在文档中注明约束。

### 4.5 其他观察（非缺陷）

- `analyze_safe_r2_*.py` 等报告的 ISC004（隐式字符串拼接）：经人工核对为**报告模板中的多行字符串**，语义正确，仅风格提示。
- 全仓库 ruff 报告约 1.1k 项，其中 800+ 为可自动修复的风格类（import 排序、旧式 typing 等）；本次改动未新增任何 lint 项（修改的三个文件：12 → 11，消除的正是 B023）。

## 五、复现方式（关键命令）

```powershell
# 1) 全量编译检查
python -c "import compileall,sys; sys.exit(0 if compileall.compile_dir(r'<repo>', quiet=1) else 1)"

# 2) 测试套件（轻量环境 = 仅 numpy/cv2/yaml/pandas/pytest）
python -m pytest tests/ -q --continue-on-collection-errors

# 3) 网格几何验证（需将官方复赛图解压至 data/semifinal/test）
#    配对规则: <prefix>_<r>_<c>.jpg (裁剪) vs <prefix>.jpg (整图)
#    期望: best_tile_idx == r*3+c 且 (0,0) 平移最优

# 4) prepare_data 端到端（真实图像 + 手写 VOC XML 小样例）
python scripts/prepare_data.py --source <src> --output <out> --tile-size 1024 --overlap 0.25 --workers 1
python scripts/prepare_data.py --source <src> --output <out2> --tile-layout semifinal_grid --workers 1
```

---

*本报告与修复分支同时提交；所有修改均附 diff，未触碰任何训练/评估行为。*
