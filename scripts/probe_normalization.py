"""Train-only diagnostic of EMA/BatchNorm; no optimizer, no source-label edits."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steel_defect.classes import CLASS_NAMES
from steel_defect.governance import digest_file
from steel_defect.image_io import imread
from steel_defect.metrics import evaluate_predictions
from steel_defect.runtime import write_manifest


def score(model, paths, truth):
    predictions, seen = [], set()
    for result in model.predict(source=[str(p.resolve()) for p in paths], imgsz=1024, batch=2,
                                conf=.001, iou=.7, max_det=300, device="0", half=False, stream=True, verbose=False):
        name = Path(result.path).name
        if name in seen or name not in truth:
            raise ValueError("Invalid scope")
        seen.add(name)
        for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            predictions.append({"image_id": name, "category_name": CLASS_NAMES[int(cls)], "bbox": box, "score": conf})
    if seen != set(truth):
        raise ValueError("Incomplete diagnostic scope")
    return evaluate_predictions(predictions, truth, CLASS_NAMES, operating_threshold=.05,
                                image_sizes={name: (1024, 1024) for name in truth})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--raw-weights", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/diagnostic_v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Probe output exists")
    args.output.mkdir(parents=True)
    data = json.loads((args.data_dir / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if not data.get("diagnostic_only") or not data.get("train_equals_validation"):
        raise ValueError("BN probe must use intentional train-only diagnostic data")
    paths, truth = [], {}
    for item in data["items"]:
        path = args.data_dir / "images" / "diagnostic" / f"{item['id']}.jpg"
        if digest_file(path) != item["image_sha256"]:
            raise ValueError("Diagnostic crop changed")
        paths.append(path)
        truth[path.name] = {}
        for class_id, box in item["annotations"]:
            truth[path.name].setdefault(CLASS_NAMES[class_id], []).append(box)
    report = {"diagnostic_only": True, "not_generalization": True,
              "weights_sha256": digest_file(args.weights), "calibration_data": str(args.data_dir),
              "calibration": "BN running-stat recalibration only, batch8, 4 shuffled passes, no optimizer, no labels used"}
    for variant, weights in [("ema_original", args.weights), ("ema_recalibrated_bn", args.weights),
                              ("raw_last", args.raw_weights)]:
        if weights is None:
            continue
        model = YOLO(str(weights))
        if list(model.names.values()) != CLASS_NAMES:
            raise ValueError("Class order mismatch")
        if variant == "ema_recalibrated_bn":
            network = model.model.float().cuda().eval()
            bn_layers = [m for m in network.modules() if isinstance(m, torch.nn.BatchNorm2d)]
            for layer in bn_layers:
                layer.reset_running_stats()
                layer.momentum = None
                layer.train()
            rng = random.Random(42)
            with torch.no_grad():
                order = paths.copy()
                for _ in range(4):
                    rng.shuffle(order)
                    for start in range(0, len(order), 8):
                        images = [imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].transpose(2, 0, 1).copy() for p in order[start:start + 8]]
                        inputs = torch.from_numpy(np.stack(images)).to("cuda", dtype=torch.float32) / 255.0
                        network(inputs)
            network.eval()
            saved = deepcopy(network).float().cpu()
            torch.save({"model": saved, "train_args": {}, "optimizer": None,
                        "diagnostic_only": True}, args.output / "bn_recalibrated_diagnostic.pt")
            del saved, inputs, images
        metrics = score(model, paths, truth)
        report[variant] = metrics
        write_manifest(args.output / "metrics.json", report)
        print(variant, json.dumps(metrics["summary"]), flush=True)
        del model
        if variant == "ema_recalibrated_bn":
            del network, bn_layers
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
