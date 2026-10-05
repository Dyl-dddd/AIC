"""Analyze quality-aware checkpoints as calibrated or localization rescue experts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_semifinal_stage1 import add_micro
from steel_defect.classes import CLASS_NAMES
from steel_defect.metrics import evaluate_predictions


VIEWS = ("original", "grid_crops")
THRESHOLDS = (
    0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.003,
    0.005, 0.007, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2,
)


def load_ground_truth(path: Path) -> tuple[dict, dict]:
    truth, sizes = {}, {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "header":
                continue
            truth[row["image_id"]] = row["ground_truth"]
            sizes[row["image_id"]] = row["image_size"]
    return truth, sizes


def summarize(predictions: list[dict], truth: dict, sizes: dict) -> dict:
    permitted = [name for name in CLASS_NAMES if name != "qilie"]
    report = add_micro(evaluate_predictions(
        predictions, truth, permitted, operating_threshold=0.0, image_sizes=sizes
    ))
    return report["summary"]


def grouped(predictions: list[dict]) -> dict[tuple[str, str], list[tuple[int, dict]]]:
    result: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        result[(prediction["image_id"], prediction["category_name"])].append(
            (index, prediction)
        )
    return result


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    upper_left = np.maximum(left[:, None, :2], right[None, :, :2])
    lower_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = np.maximum(lower_right - upper_left, 0.0)
    intersection = intersection[..., 0] * intersection[..., 1]
    left_area = np.maximum(left[:, 2] - left[:, 0], 0.0) * np.maximum(
        left[:, 3] - left[:, 1], 0.0
    )
    right_area = np.maximum(right[:, 2] - right[:, 0], 0.0) * np.maximum(
        right[:, 3] - right[:, 1], 0.0
    )
    return intersection / np.maximum(
        left_area[:, None] + right_area[None, :] - intersection, 1e-12
    )


def localization_fusion(
    control: list[dict], quality: list[dict], quality_threshold: float,
    match_iou: float, control_weight: float,
) -> tuple[list[dict], dict]:
    """Keep control membership/scores, but fuse boxes with one quality partner."""
    fused = [dict(item) for item in control]
    quality_groups = grouped([
        item for item in quality if float(item["score"]) >= quality_threshold
    ])
    control_groups = grouped(control)
    matched = 0
    ious = []
    for key, control_items in control_groups.items():
        quality_items = quality_groups.get(key, [])
        if not quality_items:
            continue
        left = np.asarray([item[1]["bbox"] for item in control_items], dtype=np.float32)
        right = np.asarray([item[1]["bbox"] for item in quality_items], dtype=np.float32)
        matrix = iou_matrix(left, right)
        best = matrix.argmax(axis=1)
        best_iou = matrix[np.arange(len(left)), best]
        for local_index, overlap in enumerate(best_iou):
            if float(overlap) < match_iou:
                continue
            control_index = control_items[local_index][0]
            quality_box = right[int(best[local_index])]
            control_box = left[local_index]
            box = control_weight * control_box + (1.0 - control_weight) * quality_box
            fused[control_index]["bbox"] = [float(value) for value in box]
            matched += 1
            ious.append(float(overlap))
    return fused, {
        "matched_boxes": matched,
        "mean_match_iou": float(np.mean(ious)) if ious else 0.0,
        "membership_unchanged": True,
    }


def metric_row(tag: str, summary: dict, **extra) -> dict:
    return {
        "tag": tag,
        "score": float(summary["score_proxy_20_60_20"]),
        "map50": float(summary["map50_macro"]),
        "map75": float(summary["map75_macro"]),
        "map50_95": float(summary["map50_95_macro"]),
        "recall": float(summary["recall_micro"]),
        "precision": float(summary["precision_micro"]),
        "false_positives_per_image": float(summary["false_positives_per_image"]),
        "export_count": int(summary["true_positives"] + summary["false_positives"]),
        **extra,
    }


def analyze_view(root: Path, view: str) -> dict:
    control_dir = root / "bce_control"
    quality_dir = root / "varifocal"
    control = json.loads((control_dir / f"bce_control_{view}_predictions.json").read_text())
    quality = json.loads((quality_dir / f"varifocal_{view}_predictions.json").read_text())
    truth, sizes = load_ground_truth(
        control_dir / f"bce_control_{view}_ground_truth.jsonl.gz"
    )
    control_summary = summarize(control, truth, sizes)
    rows = [metric_row("bce_control", control_summary)]
    for threshold in THRESHOLDS:
        filtered = [item for item in quality if float(item["score"]) >= threshold]
        rows.append(metric_row(
            f"varifocal_threshold_{threshold:g}", summarize(filtered, truth, sizes),
            threshold=threshold,
        ))
    for threshold in (0.0001, 0.001, 0.01):
        for match_iou in (0.3, 0.5, 0.7):
            for control_weight in (0.25, 0.5, 0.75):
                predictions, diagnostics = localization_fusion(
                    control, quality, threshold, match_iou, control_weight
                )
                rows.append(metric_row(
                    f"loc_fusion_q{threshold:g}_iou{match_iou:g}_cw{control_weight:g}",
                    summarize(predictions, truth, sizes), **diagnostics,
                    quality_threshold=threshold, match_iou=match_iou,
                    control_weight=control_weight,
                ))
    best_score = max(rows, key=lambda row: (row["score"], row["map50"]))
    safe_localization = [
        row for row in rows
        if row["tag"].startswith("loc_fusion")
        and row["map50"] >= control_summary["map50_macro"]
        and row["map75"] >= control_summary["map75_macro"]
        and row["map50_95"] >= control_summary["map50_95_macro"]
        and row["recall"] >= control_summary["recall_micro"] - 0.002
    ]
    return {
        "baseline": rows[0],
        "best_score": best_score,
        "best_safe_localization": max(
            safe_localization, key=lambda row: (row["score"], row["map75"]), default=None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {view: analyze_view(args.evaluation_root, view) for view in VIEWS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for view in VIEWS:
        print(view, json.dumps({
            "baseline": result[view]["baseline"],
            "best_score": result[view]["best_score"],
            "best_safe_localization": result[view]["best_safe_localization"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
