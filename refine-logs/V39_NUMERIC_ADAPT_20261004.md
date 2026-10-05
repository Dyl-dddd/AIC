# V39 numeric-source adaptation (local, 2026-10-04)

Outcome: rejected. No submission package was created.

- Starting checkpoint: `best_score_model.pt`, SHA256 `d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d`.
- Frozen split: 1,457 numeric-source train images, 295 numeric-source dev images. Derived views: 2,564 train and 2,052 dev. No Test images or labels used for fitting.
- RTX 4060 local; Ultralytics 8.3.169; YOLO11s; 1024 px; batch 1; workers 0; 5 epochs; AdamW `lr0=2e-5`, zero bias warmup. Final weights and logs in `runs/semifinal/v39_numeric_adapt/`.
- Identical dev-view protocol: source mAP50 0.357127; epoch 0 0.304206; epoch 2 0.297708; epoch 4 / last 0.302814. Best candidate delta **-0.052922**.
- Source zonglie AP50 0.352050; best candidate epoch 0 0.237259. Source qilie AP50 0; epoch 0 0.006072, epoch 2 0.003302, last 0. The tiny q gain is far smaller than the other class losses.
- The numeric-source train set has only seven qilie boxes, all narrow (19–66 px wide), whereas numeric dev has eight much broader qilie boxes (124–479 px) in a single image. The morphology mismatch limits this strategy.
- This dev protocol is not the official competition score. Protected official best remains V12 = 67.92; V39 is not submitted.

Source report: `runs/semifinal/v39_numeric_adapt/RESULT.json`.
