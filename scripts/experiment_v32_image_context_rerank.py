"""Test a supervised image-context prior against frozen V12 OOF predictions.

The classifier is fit only on the official numeric-source training split.
The independent numeric-source dev split selects whether a class prior is safe.
No Test labels are loaded; this script does not create a submission.
"""

from __future__ import annotations

import base64
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
from steel_defect.metrics import _match_class, average_precision

SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
PRED_ROOT = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
CELL_ROOT = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OUT = ROOT / "runs/semifinal/v32_image_context_20261003"
CLASSES = ("jieba", "zonglie", "jiaza", "yiwuyaru", "huashang",
           "mamianmakeng", "yanghuatiepi", "gunyin")
WEIGHTS = (0.25, 0.5, 1.0)


def image_features(gray: np.ndarray) -> np.ndarray:
    """Small, fixed texture + spatial features from the exact 64x64 audit thumbnail."""
    a = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    lap = cv2.Laplacian(a, cv2.CV_32F)
    mag = cv2.magnitude(gx, gy)
    arrays = (a, np.abs(gx), np.abs(gy), np.abs(lap), mag)
    features = []
    for z in arrays:
        features.extend(np.percentile(z, [0, 5, 25, 50, 75, 95, 99, 100]).tolist())
        features.extend([float(z.mean()), float(z.std())])
        # Fixed-grid pooling captures defect location without a high-dimensional raw image.
        features.extend(cv2.resize(z, (4, 4), interpolation=cv2.INTER_AREA).ravel().tolist())
        features.extend(cv2.resize(z, (2, 2), interpolation=cv2.INTER_AREA).ravel().tolist())
    for axis in (0, 1):
        row_mean = a.mean(axis=axis)
        row_std = a.std(axis=axis)
        features.extend([float(row_mean.std()), float(row_std.mean()),
                         float(row_std.std()), float(np.max(row_std))])
    features.extend([float(np.mean(np.abs(gx) > np.abs(gy))),
                     float(np.mean(mag > 0.08)), float(np.mean(mag > 0.16))])
    return np.asarray(features, dtype=np.float32)


def ap(rows: list[dict], gt: dict, name: str) -> float:
    tp, fp, count = _match_class(rows, gt, name, 0.5)
    return average_precision(tp, fp, count)


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    records = split["records"]
    train_names = [n for n in split["splits"]["train"] if not n.startswith("C")]
    dev_names = [n for n in split["splits"]["dev"] if not n.startswith("C")]
    all_names = train_names + dev_names
    features = np.stack([
        image_features(np.frombuffer(base64.b64decode(records[n]["thumbnail"]), np.uint8).reshape(64, 64))
        for n in all_names
    ])
    n_train = len(train_names)
    probabilities: dict[str, dict[str, float]] = {}
    image_quality = {}
    for name in ("any",) + CLASSES:
        labels = np.asarray([
            bool(records[n]["annotations"]) if name == "any" else
            any(item[0] == name for item in records[n]["annotations"])
            for n in all_names
        ], dtype=np.int32)
        pos = int(labels[:n_train].sum())
        if pos < 15:
            continue
        model = XGBClassifier(
            n_estimators=180, max_depth=2, learning_rate=0.035,
            min_child_weight=12 if pos >= 100 else 5,
            subsample=0.85, colsample_bytree=0.8, reg_lambda=12,
            objective="binary:logistic", eval_metric="logloss", n_jobs=4,
            random_state=20261003,
            scale_pos_weight=min(8.0, (n_train - pos) / pos),
        )
        model.fit(features[:n_train], labels[:n_train])
        p = model.predict_proba(features[n_train:])[:, 1]
        image_quality[name] = {
            "train_positive": pos, "dev_positive": int(labels[n_train:].sum()),
            "dev_image_ap": float(average_precision_score(labels[n_train:], p)),
            "dev_image_auc": float(roc_auc_score(labels[n_train:], p)),
        }
        probabilities[name] = dict(zip(dev_names, map(float, p), strict=True))
        print(name, image_quality[name], flush=True)

    gt_by_view = {}
    rows_by_view = {}
    for view in ("original", "grid_crops"):
        gt = load_cell(CELL_ROOT / f"model_a_tta_s1024_{view}")["ground_truth"]
        gt_by_view[view] = {n: labels for n, labels in gt.items()
                            if n.split("::", 1)[0] in probabilities["any"]}
        rows = json.loads((PRED_ROOT / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        rows_by_view[view] = [r for r in rows
                              if r["image_id"].split("::", 1)[0] in probabilities["any"]]
    report = {"image_quality": image_quality, "classes": {}, "protocol": {
        "train_images": len(train_names), "dev_images": len(dev_names),
        "views": ["original", "grid_crops"], "weights": WEIGHTS,
        "score_rule": "score * image_probability ** weight, within each class",
    }}
    for name in CLASSES:
        if name not in probabilities:
            continue
        class_report = {}
        for view in ("original", "grid_crops"):
            gt = gt_by_view[view]
            rows = [r for r in rows_by_view[view] if r["category_name"] == name]
            base = ap(rows, gt, name)
            trials = {"baseline": base}
            for target in ("any", name):
                if target not in probabilities:
                    continue
                for w in WEIGHTS:
                    rescored = [{**r, "score": float(r["score"]) *
                                 max(1e-4, probabilities[target][r["image_id"].split("::", 1)[0]]) ** w}
                                for r in rows]
                    trials[f"{target}_{w}"] = ap(rescored, gt, name)
            class_report[view] = trials
        report["classes"][name] = class_report
        best = max((k for k in class_report["original"] if k != "baseline"),
                   key=lambda k: .6193 * class_report["original"][k] +
                   .3807 * class_report["grid_crops"][k])
        print(name, "baseline", *(round(class_report[v]["baseline"], 5)
                                  for v in ("original", "grid_crops")),
              "best", best, *(round(class_report[v][best], 5)
                               for v in ("original", "grid_crops")), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Report:", OUT / "report.json", flush=True)


if __name__ == "__main__":
    main()
