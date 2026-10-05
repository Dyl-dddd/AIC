"""Add train-only YOLO false-positive crops to the independent verifier dataset.

The frozen development split and Test split are never used for mining or fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts import v20_crop_verifier as verifier


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/semifinal/v20_crop_verifier_20261002"
MODEL = ROOT / "analysis/ensemble_v4_results_20260927/runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--per-image", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.0001)
    args = parser.parse_args()
    out = verifier.OUT
    if out.resolve() == BASE.resolve():
        raise ValueError("set CROP_VERIFIER_OUT to a new v21 experiment directory")
    metadata = json.loads(verifier.SPLITS.read_text(encoding="utf-8"))
    train_names = metadata["splits"]["train"]
    empty = [name for name in train_names if not metadata["records"][name]["annotations"]]
    nonempty = [name for name in train_names if metadata["records"][name]["annotations"]]
    # Deterministic mix of easy-to-label empty images and nonempty images with
    # detector proposals that are far from every annotated defect.
    selected = empty[::max(1, len(empty) // max(1, args.limit // 2))][:args.limit // 2]
    selected += nonempty[::max(1, len(nonempty) // max(1, args.limit - len(selected)))][:args.limit - len(selected)]
    original = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    manifest = [{**row, "file": str((BASE / "crops" / row["file"]).resolve())} for row in original]
    crop_root = out / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    counts = Counter()
    for index, name in enumerate(selected, 1):
        gray = verifier.gray_image(verifier.IMAGES / name)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        result = model.predict(rgb, imgsz=args.imgsz, conf=args.conf, iou=0.7,
                               max_det=300, device=0, verbose=False)[0]
        annotations = metadata["records"][name]["annotations"]
        gt = [list(map(float, row[1:])) for row in annotations]
        group = metadata["records"][name]["group"]
        group_hash = hashlib.sha256(str(group).encode()).digest()
        split = "val" if int.from_bytes(group_hash[:4], "big") % 10 == 0 else "train"
        candidates = []
        for box, score, class_id in zip(result.boxes.xyxy.cpu().numpy(),
                                        result.boxes.conf.cpu().numpy(),
                                        result.boxes.cls.int().cpu().numpy(), strict=True):
            if model.names[int(class_id)] not in verifier.CLASSES[1:]:
                continue
            coordinates = [float(value) for value in box]
            if verifier.iou_one(coordinates, gt) >= 0.05:
                continue
            candidates.append((float(score), int(class_id), coordinates))
        candidates.sort(reverse=True)
        per_class = Counter()
        chosen = []
        for score, class_id, coordinates in candidates:
            if per_class[class_id] >= 4:
                continue
            chosen.append((score, class_id, coordinates))
            per_class[class_id] += 1
            if len(chosen) >= args.per_image:
                break
        for local_index, (score, class_id, coordinates) in enumerate(chosen):
            crop = verifier.crop_patch(gray, coordinates)
            destination = crop_root / f"hard_{index:05d}_{local_index:02d}.jpg"
            ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError(destination)
            buffer.tofile(str(destination))
            manifest.append({"file": str(destination.resolve()), "label": 0,
                             "split": split, "source": name, "detector_score": score,
                             "detector_class": model.names[class_id]})
            counts[split] += 1
        if index % 50 == 0:
            print(f"MINE {index}/{len(selected)} added={sum(counts.values())}", flush=True)
    if not counts:
        raise RuntimeError("no hard negatives mined")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    report = {"source_train_images": len(selected), "source_dev_images_used": 0,
              "source_test_images_used": 0, "hard_negative_counts": dict(counts),
              "total_samples": len(manifest), "detector": str(MODEL), "imgsz": args.imgsz,
              "per_image": args.per_image, "conf": args.conf}
    (out / "mining_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
