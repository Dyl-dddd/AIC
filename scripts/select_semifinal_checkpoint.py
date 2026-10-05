"""Evaluate saved fine-tuning checkpoints every five epochs on crop-style dev.

This is a cheap post-training selection pass. Training itself keeps validation
disabled to avoid the host-memory accumulation seen in earlier runs.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re

import numpy as np
import torch
from ultralytics import YOLO


def checkpoint_order(path: Path) -> tuple[int, str]:
    match = re.search(r"epoch(\d+)$", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a semifinal fine-tuning checkpoint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--extra-weights", type=Path, nargs="*", default=[],
                        help="Additional checkpoints, e.g. the pretrained baseline")
    args = parser.parse_args()

    weights_dir = args.run_dir / "weights"
    checkpoints = sorted(weights_dir.glob("epoch*.pt"), key=checkpoint_order)
    for checkpoint in (weights_dir / "best.pt", weights_dir / "last.pt", *args.extra_weights):
        if checkpoint.is_file() and checkpoint not in checkpoints:
            checkpoints.append(checkpoint)
    if not checkpoints:
        raise SystemExit(f"No epoch*.pt or last.pt checkpoints found under {weights_dir}")

    rows = []
    for checkpoint in checkpoints:
        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(args.data),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            conf=0.001,
            iou=0.70,
            max_det=args.max_det,
            half=not str(args.device).lower().startswith("cpu"),
            plots=False,
            save_json=False,
            verbose=False,
            project=str(args.run_dir / "checkpoint_eval"),
            name=checkpoint.stem,
            exist_ok=True,
        )
        recall = [float(value) for value in np.asarray(metrics.box.r).reshape(-1)]
        row = {
            "checkpoint": str(checkpoint.resolve()),
            "recall_macro": float(np.mean(recall)) if recall else 0.0,
            "map50_macro": float(metrics.box.map50),
            "map50_95_macro": float(metrics.box.map),
            "per_class_recall": recall,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        del metrics, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (row["recall_macro"], row["map50_macro"]), reverse=True)
    report = {
        "selection_rule": "maximize macro recall; break ties with macro mAP50",
        "best": rows[0],
        "ranked": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {rows[0]['checkpoint']}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
