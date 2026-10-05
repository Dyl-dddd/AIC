# V14 V13-evidence hybrid ranker（RTX 4090）

本包把 V13 定位专家作为候选质量证据，但保持 V7/V12 的候选框成员和坐标完全不变。严格五折官方组 OOF 显示：只对 `jieba`、`jiaza`、`mamianmakeng` 启用 V13 增强模型，其余类别保留已官方验证的 V12 模型，是双视图更稳健的方案。

开发集代理变化（不是官方分数）：

- 相对 V7：original `+0.695806`，grid-crops `+0.790852`，加权 `+0.731991`。
- 相对 V12：original `+0.074322`，grid-crops `+0.092296`。
- 候选数量、类别、框坐标全部保持不变；只重排 score。

## 运行

```bash
cd /mnt/proj/iron
unzip -qo semifinal_v14_v13_hybrid_ranker_20261001.zip
cd semifinal_v14_v13_hybrid_ranker_20261001

python -m pip install --no-cache-dir -r requirements-v14.txt

python -u generate_v14_v13_hybrid_ranker_submission.py \
  --project-root /mnt/proj/iron \
  --v10-root /mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930 \
  --preflight-only

python -u generate_v14_v13_hybrid_ranker_submission.py \
  --project-root /mnt/proj/iron \
  --v10-root /mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930
```

程序会先用 V13 定位专家在 1536 分辨率对 788 张 Test 图运行一次可恢复推理，再复用 V7、Varifocal、V9-P1/P2 缓存做逐类排序。预计约 1–3 小时。

## 提交文件

只提交：

```text
/mnt/proj/iron/runs/semifinal/v14_v13_hybrid_ranker_submission/semifinal_v14_v13_hybrid_ranker_SUBMIT_ONLY.zip
```

该 ZIP 内严格只有一个 `submission.json`。不要提交本代码包，也不要提交中间的 V13 预测 JSON。

## 必需的既有文件

- V7 primary submission JSON
- Varifocal Test prediction cache
- V9 P1/P2 Test prediction caches
- V13 localization checkpoint：`runs/semifinal/v13_hires_localize_vfl/weights/last.pt`
- 已解压的 V10 fix2 代码目录

预检会逐项检查这些文件、V13 权重 SHA-256、RTX 4090、Ultralytics 版本、788 张图片和包内清单；缺失时会在 GPU 推理前停止。
