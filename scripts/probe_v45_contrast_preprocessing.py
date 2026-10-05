"""Paired, read-only V12-source preprocessing probe on frozen numeric dev tiles.

The only intervention is a per-image 1.25x contrast boost about the steel-pixel
median. Black padding, image dimensions, labels, and train/test membership stay
unchanged. This is a directional detector probe, not the V12 ranked submission.
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
from steel_defect.metrics import _match_class, average_precision

MODEL = ROOT / "analysis/full_retrain_v3_fix1_20260925/runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt"
MODEL_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
IMAGES = ROOT / "runs/semifinal/v39_numeric_adapt/dataset/images/val"
LABELS = ROOT / "runs/semifinal/v39_numeric_adapt/dataset/labels/val"
OUT = ROOT / "runs/semifinal/v45_contrast_preprocessing_20261005"
CLASSES = ("jieba", "zonglie", "qilie", "jiaza", "yiwuyaru", "huashang", "mamianmakeng", "yanghuatiepi", "gunyin")


def contrast_boost(image: np.ndarray, factor: float = 1.25) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    steel = gray > 5
    if steel.sum() < 100:
        return image.copy()
    center = float(np.median(gray[steel]))
    shifted = center + factor * (image.astype(np.float32) - center)
    result = np.clip(shifted, 0, 255).astype(np.uint8)
    result[~steel] = image[~steel]
    return result


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode {path}")
    return image


def read_labels(path: Path, width: int, height: int) -> dict:
    result = defaultdict(list)
    if not path.exists():
        raise RuntimeError(f"Missing label file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        result[CLASSES[class_id]].append([
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ])
    return dict(result)


def summarize(rows: list[dict], gt: dict) -> dict:
    per_class = {}
    for name in CLASSES:
        tp, fp, count = _match_class(rows, gt, name, .5)
        per_class[name] = {
            "gt": int(count), "tp": int(tp.sum()), "fp": int(fp.sum()),
            "ap50": average_precision(tp, fp, count) if count else None,
            "top100_tp": int(tp[:100].sum()),
        }
    ap8 = [entry["ap50"] for name, entry in per_class.items()
           if name != "qilie" and entry["ap50"] is not None]
    ap9 = [entry["ap50"] for entry in per_class.values() if entry["ap50"] is not None]
    return {
        "predictions": len(rows), "gt": sum(entry["gt"] for entry in per_class.values()),
        "macro_ap50_8_without_qilie": float(np.mean(ap8)) if ap8 else None,
        "macro_ap50_9": float(np.mean(ap9)) if ap9 else None,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; not a release gate")
    parser.add_argument("--imgsz", type=int, default=1024)
    args = parser.parse_args()
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("V12 source checkpoint SHA256 mismatch")
    files = sorted(IMAGES.glob("*__grid*.jpg"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("Frozen numeric development tiles are missing")
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / ("smoke" if args.limit else "full")
    cache.mkdir(exist_ok=True)
    model = None
    ground_truth = {}
    predictions = {"plain": [], "boost_125": []}
    pixel_stats = {"plain_std": [], "boost_std": []}
    for index, path in enumerate(files, 1):
        original = read_image(path)
        modified = contrast_boost(original)
        if original.shape != modified.shape:
            raise RuntimeError("Photometric transform changed image geometry")
        height, width = original.shape[:2]
        ground_truth[path.name] = read_labels(LABELS / path.with_suffix(".txt").name, width, height)
        pixel_stats["plain_std"].append(float(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).std()))
        pixel_stats["boost_std"].append(float(cv2.cvtColor(modified, cv2.COLOR_BGR2GRAY).std()))
        target = cache / (path.stem + ".json")
        if target.exists():
            pair = json.loads(target.read_text(encoding="utf-8"))
        else:
            if model is None:
                model = YOLO(str(MODEL))
                if tuple(model.names[i] for i in range(len(CLASSES))) != CLASSES:
                    raise RuntimeError("Source checkpoint class order differs from frozen labels")
            outputs = model.predict([original, modified], imgsz=args.imgsz, batch=2,
                                    conf=.001, iou=.7, max_det=1000, device=0,
                                    half=True, verbose=False)
            pair = {}
            for name, output in zip(("plain", "boost_125"), outputs, strict=True):
                boxes = output.boxes
                pair[name] = [
                    {"image_id": path.name, "category_name": CLASSES[int(cid)],
                     "bbox": [float(v) for v in xyxy], "score": float(score)}
                    for xyxy, score, cid in zip(boxes.xyxy.cpu().numpy(),
                                                boxes.conf.cpu().numpy(),
                                                boxes.cls.cpu().numpy(), strict=True)
                ] if boxes is not None else []
            target.write_text(json.dumps(pair, separators=(",", ":")), encoding="utf-8")
        for key in predictions:
            predictions[key].extend(pair[key])
        if index % 100 == 0 or index == len(files):
            print(f"PAIRED {index}/{len(files)}", flush=True)
    report = {
        "warning": "Frozen numeric development tiles only; source detector, not full V12 ranker or official Test. No score increase may be inferred from this probe alone.",
        "images": len(files), "source_images": len({path.stem.split("__grid")[0] for path in files}),
        "model_sha256": MODEL_SHA256, "imgsz": args.imgsz,
        "transform": "Steel-pixel (>5 gray) median-centered contrast x1.25; black padding unchanged; labels unchanged",
        "pixel_std_median": {key: float(np.median(values)) for key, values in pixel_stats.items()},
        "plain": summarize(predictions["plain"], ground_truth),
        "boost_125": summarize(predictions["boost_125"], ground_truth),
    }
    target = OUT / ("smoke_report.json" if args.limit else "report.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT {target}", flush=True)
    print(f"AP8 plain={report['plain']['macro_ap50_8_without_qilie']}; "
          f"boost={report['boost_125']['macro_ap50_8_without_qilie']}", flush=True)


if __name__ == "__main__":
    main()
