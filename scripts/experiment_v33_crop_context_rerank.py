"""Independent crop-context check for the V32 mamianmakeng signal.

Fits on six numeric-source crops per training image, evaluates on numeric-source
dev crops and frozen V12 OOF boxes. No Test labels or submissions are used.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell
from scripts.experiment_v32_image_context_rerank import image_features
from steel_defect.metrics import _match_class, average_precision

SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
PRED_ROOT = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
CELL_ROOT = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OUT = ROOT / "runs/semifinal/v33_crop_context_20261003"
CLASS = "mamianmakeng"
XS = (0, 1354, 2709)
YS = (0, 1484)
WIDTH, HEIGHT = 1387, 1516


def crop_rows(names: list[str], records: dict, gt: dict | None = None):
    ids, features, labels = [], [], []
    for index, name in enumerate(names, 1):
        path = ROOT / "data/official/train" / name
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(name)
        for y in YS:
            for x in XS:
                crop = image[y:y + HEIGHT, x:x + WIDTH]
                if crop.shape != (HEIGHT, WIDTH):
                    raise RuntimeError((name, crop.shape))
                crop_id = f"{name}::x{x}_y{y}"
                ids.append(crop_id)
                features.append(image_features(cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)))
                if gt is None:
                    anns = records[name]["annotations"]
                    # A crop is positive when a ground-truth defect has meaningful visible area.
                    labels.append(int(any(
                        a[0] == CLASS and
                        max(0.0, min(float(a[3]), x + WIDTH) - max(float(a[1]), x)) *
                        max(0.0, min(float(a[4]), y + HEIGHT) - max(float(a[2]), y)) >=
                        0.3 * (float(a[3]) - float(a[1])) * (float(a[4]) - float(a[2]))
                        for a in anns
                    )))
                else:
                    labels.append(int(bool(gt[crop_id].get(CLASS))))
        if index % 250 == 0:
            print("feature images", index, "/", len(names), flush=True)
    return ids, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def ap(rows: list[dict], gt: dict) -> float:
    tp, fp, count = _match_class(rows, gt, CLASS, 0.5)
    return average_precision(tp, fp, count)


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = split["records"]
    train_names = [n for n in split["splits"]["train"] if not n.startswith("C")]
    dev_names = [n for n in split["splits"]["dev"] if not n.startswith("C")]
    cell = load_cell(CELL_ROOT / "model_a_tta_s1024_grid_crops")
    all_gt = cell["ground_truth"]
    OUT.mkdir(parents=True, exist_ok=True)
    train_cache = OUT / "train_features.npz"
    dev_cache = OUT / "dev_features.npz"
    if train_cache.exists():
        data = np.load(train_cache)
        train_ids, train_x, train_y = data["ids"].tolist(), data["x"], data["y"]
    else:
        train_ids, train_x, train_y = crop_rows(train_names, records)
        np.savez_compressed(train_cache, ids=train_ids, x=train_x, y=train_y)
    if dev_cache.exists():
        data = np.load(dev_cache)
        dev_ids, dev_x, dev_y = data["ids"].tolist(), data["x"], data["y"]
    else:
        dev_ids, dev_x, dev_y = crop_rows(dev_names, records, all_gt)
        np.savez_compressed(dev_cache, ids=dev_ids, x=dev_x, y=dev_y)
    pos = int(train_y.sum())
    model = XGBClassifier(n_estimators=180, max_depth=2, learning_rate=0.035,
                          min_child_weight=12, subsample=0.85, colsample_bytree=0.8,
                          reg_lambda=12, objective="binary:logistic", eval_metric="logloss",
                          n_jobs=4, random_state=20261003,
                          scale_pos_weight=min(8.0, (len(train_y) - pos) / pos))
    model.fit(train_x, train_y)
    probs = dict(zip(dev_ids, map(float, model.predict_proba(dev_x)[:, 1]), strict=True))
    image_metrics = {"train_crop_positive": pos, "dev_crop_positive": int(dev_y.sum()),
                     "dev_crop_ap": float(average_precision_score(dev_y, list(probs.values()))),
                     "dev_crop_auc": float(roc_auc_score(dev_y, list(probs.values())))}
    all_rows = json.loads((PRED_ROOT / "v12_grid_crops_predictions.json").read_text(encoding="utf-8"))
    rows = [r for r in all_rows if r["category_name"] == CLASS and r["image_id"] in probs]
    gt = {n: all_gt[n] for n in dev_ids}
    baseline = ap(rows, gt)
    trials = {}
    for weight in (0.1, 0.25, 0.5, 1.0):
        rescored = [{**r, "score": float(r["score"]) * max(1e-4, probs[r["image_id"]]) ** weight}
                    for r in rows]
        trials[str(weight)] = ap(rescored, gt)
    result = {"image_metrics": image_metrics, "baseline_ap50": baseline,
              "reranked_ap50": trials, "dev_crops": len(dev_ids), "predictions": len(rows)}
    (OUT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
