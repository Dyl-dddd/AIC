"""Same-image V34 zonglie complement probe against protected V12 dev OOF.

This is development inference only. It never reads Test or emits a submission.
Per-image caches allow safe restart after an interrupted GPU process.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steel_defect.geometry import box_iou
from steel_defect.metrics import _match_class, average_precision


WEIGHT = ROOT / "runs/semifinal/v41_v34_probe_20261004/v34_selected_single_detector.pt"
FROZEN = ROOT / "data/official_v2_20260831b/splits.json"
V12 = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_original_predictions.json"
OUT = ROOT / "runs/semifinal/v41_v34_probe_20261004"
CLASS = "zonglie"
CLASS_ID = 1


def read_gray(path: Path) -> np.ndarray:
    array = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if array is None:
        raise RuntimeError(f"Cannot read {path}")
    return array


def views(width: int, height: int) -> list[tuple[int, int, int, int]]:
    side, step = 1024, 768
    xs = list(range(0, max(1, width - side + 1), step))
    ys = list(range(0, max(1, height - side + 1), step))
    if xs[-1] != width - side:
        xs.append(width - side)
    if ys[-1] != height - side:
        ys.append(height - side)
    square = [(x, y, x + side, y + side) for y in ys for x in xs]
    long_strip = [(x, 0, x + side, height) for x in xs]
    return square + long_strip


def infer_one(model: YOLO, image: np.ndarray, imgsz: int) -> list[dict]:
    height, width = image.shape
    windows = views(width, height)
    candidates: list[tuple[float, list[float]]] = []
    for start in range(0, len(windows), 2):
        group = windows[start:start + 2]
        crops = [cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
                 for x1, y1, x2, y2 in group]
        results = model.predict(crops, imgsz=imgsz, conf=0.0001, iou=0.7,
                                max_det=1000, classes=[CLASS_ID], device=0,
                                half=True, verbose=False)
        for result, (left, top, _, _) in zip(results, group, strict=True):
            if result.boxes is None:
                continue
            for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(), strict=True):
                x1, y1, x2, y2 = map(float, xyxy)
                box = [max(0.0, x1 + left), max(0.0, y1 + top),
                       min(float(width), x2 + left), min(float(height), y2 + top)]
                if box[2] > box[0] + 1 and box[3] > box[1] + 1:
                    candidates.append((float(conf), box))
    if not candidates:
        return []
    rectangles = [[x1, y1, x2 - x1, y2 - y1] for _, (x1, y1, x2, y2) in candidates]
    selected = cv2.dnn.NMSBoxes(rectangles, [score for score, _ in candidates],
                                score_threshold=0.0, nms_threshold=0.5)
    indexes = np.asarray(selected, dtype=int).reshape(-1) if selected is not None else np.empty(0, dtype=int)
    return [{"category_name": CLASS, "bbox": [int(round(v)) for v in candidates[i][1]],
             "score": candidates[i][0]} for i in indexes]


def max_iou(predictions: list[dict], box: list[float], floor: float) -> float:
    selected = [item["bbox"] for item in predictions if item["score"] >= floor]
    return float(np.max(box_iou(np.asarray(box, dtype=np.float32),
                                np.asarray(selected, dtype=np.float32)))) if selected else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; not a release decision")
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    signature = hashlib.sha256(WEIGHT.read_bytes()).hexdigest()
    if signature != "e21c804001c750a3d41074628bca76fe32869cd3cae5ee422570c0294c7171c4":
        raise RuntimeError("V34 checkpoint hash mismatch")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    names = sorted(name for name in frozen["splits"]["dev"] if not name.startswith("C"))
    if args.limit:
        names = names[:args.limit]
    v12 = [row for row in json.loads(V12.read_text(encoding="utf-8"))
           if row["category_name"] == CLASS and row["image_id"] in set(names)]
    by_v12 = defaultdict(list)
    for row in v12:
        by_v12[row["image_id"]].append(row)
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    model = None
    predictions = []
    for index, name in enumerate(names, 1):
        path = cache / f"{Path(name).stem}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        if data is None or data.get("weight_sha256") != signature or data.get("imgsz") != args.imgsz:
            if model is None:
                model = YOLO(str(WEIGHT))
            rows = infer_one(model, read_gray(ROOT / "data/official/train" / name), args.imgsz)
            data = {"weight_sha256": signature, "imgsz": args.imgsz, "image_id": name,
                    "predictions": rows}
            path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        elif data.get("image_id") != name:
            raise RuntimeError(f"Cache collision: {path}")
        predictions.extend({"image_id": name, **row} for row in data["predictions"])
        if index % 10 == 0 or index == len(names):
            print(f"INFER {index}/{len(names)}: {len(predictions)} cumulative boxes", flush=True)
    gt = {}
    for name in names:
        boxes = [[float(v) for v in ann[1:]] for ann in frozen["records"][name]["annotations"]
                 if ann[0] == CLASS]
        gt[name] = {CLASS: boxes}
    eval_rows = {}
    for label, rows in (("v12", v12), ("v34", predictions)):
        tp, fp, count = _match_class(rows, gt, CLASS, 0.5)
        eval_rows[label] = {"tp": int(tp.sum()), "fp": int(fp.sum()), "gt": count,
                            "ap50": average_precision(tp, fp, count), "rows": len(rows)}
    unique = []
    for name in names:
        for box in gt[name][CLASS]:
            a = max_iou(by_v12[name], box, 0.05)
            b = max_iou([row for row in predictions if row["image_id"] == name], box, 0.05)
            if b >= 0.5 and a < 0.5:
                unique.append({"image_id": name, "gt": box, "v12_max_iou": a,
                               "v34_max_iou": b})
    report = {"warning": "Numeric-source frozen dev only; not official Test score",
              "images": len(names), "imgsz": args.imgsz, "weight_sha256": signature,
              "metrics": eval_rows, "v34_unique_gt_at_005": unique,
              "v34_unique_count": len(unique)}
    target = OUT / ("smoke.json" if args.limit else "numeric_dev_report.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {target}")
    print(f"V12 zonglie AP50={eval_rows['v12']['ap50']:.5f}; "
          f"V34={eval_rows['v34']['ap50']:.5f}; unique={len(unique)}", flush=True)


if __name__ == "__main__":
    main()
