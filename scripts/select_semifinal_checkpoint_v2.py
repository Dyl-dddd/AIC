"""Conservatively select one checkpoint against a frozen epoch25 baseline.

This is development-set screening, not an estimate of the official test score.
The semifinal submission must use one selected checkpoint only.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re

import numpy as np
import torch
import yaml
from ultralytics import YOLO


EXCLUDED = {"qilie"}  # Not a permitted semifinal submission category.


def epoch_order(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"epoch(\d+)", path.stem)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def supported_class_names(data: Path) -> list[str]:
    payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        ordered = [str(names[key]) for key in sorted(names, key=int)]
    else:
        ordered = [str(name) for name in names]
    return [name for name in ordered if name not in EXCLUDED]


def summarize_metrics(metrics, data: Path) -> dict:
    names = supported_class_names(data)
    metric_names = metrics.names
    metric_indices = np.asarray(metrics.box.ap_class_index, dtype=int).reshape(-1)
    recall = np.asarray(metrics.box.r, dtype=float).reshape(-1)
    ap50 = np.asarray(metrics.box.ap50, dtype=float).reshape(-1)
    ap = np.asarray(metrics.box.ap, dtype=float)
    if ap.ndim == 1:
        ap = ap.reshape(-1, 1)
    per_class = {}
    for pos, index in enumerate(metric_indices):
        name = str(metric_names[int(index)])
        if name in names:
            per_class[name] = {
                "recall": float(recall[pos]),
                "ap50": float(ap50[pos]),
                "ap50_95": float(np.mean(ap[pos])),
            }
    # An expected class missing from the metric object contributes zero,
    # rather than silently making the macro mean look better.
    for name in names:
        per_class.setdefault(name, {"recall": 0.0, "ap50": 0.0, "ap50_95": 0.0})
    return {
        "data": str(data.resolve()),
        "classes": names,
        "recall_macro": float(np.mean([per_class[name]["recall"] for name in names])),
        "map50_macro": float(np.mean([per_class[name]["ap50"] for name in names])),
        "map50_95_macro": float(np.mean([per_class[name]["ap50_95"] for name in names])),
        "per_class": per_class,
    }


def eligible(candidate: dict, baseline: dict, min_map50_gain: float, max_recall_drop: float) -> bool:
    return all(
        candidate[key]["map50_macro"] >= baseline[key]["map50_macro"] + min_map50_gain
        and candidate[key]["recall_macro"] >= baseline[key]["recall_macro"] - max_recall_drop
        for key in baseline
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--secondary-data", type=Path, help="Optional independent original-image dev dataset")
    parser.add_argument("--baseline", type=Path, default=Path("best_score_proxy_epoch25.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--min-map50-gain", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    args = parser.parse_args()
    if not args.baseline.is_file():
        parser.error(f"Baseline checkpoint missing: {args.baseline}")
    data_paths = {"crop_dev": args.data}
    if args.secondary_data:
        if not args.secondary_data.is_file():
            parser.error(f"Secondary dataset YAML missing: {args.secondary_data}")
        data_paths["secondary_dev"] = args.secondary_data
    for data in data_paths.values():
        if not data.is_file():
            parser.error(f"Dataset YAML missing: {data}")
    weights_dir = args.run_dir / "weights"
    checkpoints = sorted(weights_dir.glob("epoch*.pt"), key=epoch_order)
    for path in (weights_dir / "best.pt", weights_dir / "last.pt"):
        if path.is_file() and path not in checkpoints:
            checkpoints.append(path)
    if not checkpoints:
        parser.error(f"No checkpoints under {weights_dir}")

    rows = []
    for checkpoint in (args.baseline, *checkpoints):
        model = YOLO(str(checkpoint))
        evaluations = {}
        for key, data in data_paths.items():
            metrics = model.val(
                data=str(data), split="val", imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=args.device, conf=0.001, iou=0.70,
                max_det=args.max_det, half=not str(args.device).lower().startswith("cpu"),
                plots=False, save_json=False, verbose=False,
                project=str(args.run_dir / "checkpoint_eval_v2"),
                name=f"{checkpoint.stem}_{key}", exist_ok=True,
            )
            evaluations[key] = summarize_metrics(metrics, data)
            del metrics
        row = {"checkpoint": str(checkpoint.resolve()), "evaluations": evaluations}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = rows[0]
    passing = [
        row for row in rows[1:]
        if eligible(row["evaluations"], baseline["evaluations"], args.min_map50_gain, args.max_recall_drop)
    ]
    passing.sort(
        key=lambda row: (
            row["evaluations"]["crop_dev"]["map50_macro"],
            row["evaluations"]["crop_dev"]["recall_macro"],
        ),
        reverse=True,
    )
    selected = passing[0] if passing else baseline
    report = {
        "selection_rule": "one checkpoint; improve permitted-class mAP50 >= baseline + margin and keep recall >= baseline - tolerance on every dev view",
        "min_map50_gain": args.min_map50_gain,
        "max_recall_drop": args.max_recall_drop,
        "selected": selected,
        "new_checkpoint_passed": bool(passing),
        "baseline": baseline,
        "candidates": rows[1:],
        "warning": "Development-set screen only; official semifinal score is unknown. Do not ensemble training stages.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected: {selected['checkpoint']}")
    print(f"New checkpoint passed: {bool(passing)}")
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
