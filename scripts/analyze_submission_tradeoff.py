"""Evaluate score ranking and threshold tradeoffs on a labeled dev split.

The input must be grouped by image, as produced by build_layered_submission.py.
No test-set labels or platform scores are used.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_recall_submission import load_ground_truth
from scripts.validate_large_submission import iter_json_array
from steel_defect.classes import CLASS_NAMES


THRESHOLDS = (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1)
TOP_K_VALUES = (25, 50, 100, 200, 500, 1000, 2000)


def ap50_from_ranked_labels(scores: np.ndarray, labels: np.ndarray, gt_count: int) -> float:
    """101-point AP, equivalent to the repository metric without full FP arrays."""
    if gt_count <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    positive_ranks = np.flatnonzero(labels[order]) + 1
    if not len(positive_ranks):
        return 0.0
    precision_at_hit = np.arange(1, len(positive_ranks) + 1) / positive_ranks
    envelope = np.maximum.accumulate(precision_at_hit[::-1])[::-1]
    required = np.maximum(1, np.ceil(np.linspace(0, 1, 101) * gt_count).astype(int))
    values = np.zeros(101, dtype=np.float64)
    valid = required <= len(envelope)
    values[valid] = envelope[required[valid] - 1]
    return float(values.mean())


def label_image_class(predictions: list[dict], gt_boxes: list[list[float]]) -> np.ndarray:
    """Greedy score-ordered 1:1 IoU>=0.5 labels for one image and class."""
    labels = np.zeros(len(predictions), dtype=np.uint8)
    if not predictions or not gt_boxes:
        return labels
    boxes = np.asarray([item["bbox"] for item in predictions], dtype=np.float32)
    truth = np.asarray(gt_boxes, dtype=np.float32)
    intersection_min = np.maximum(boxes[:, None, :2], truth[None, :, :2])
    intersection_max = np.minimum(boxes[:, None, 2:], truth[None, :, 2:])
    intersection_size = np.maximum(0, intersection_max - intersection_min)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    box_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    truth_area = np.prod(truth[:, 2:] - truth[:, :2], axis=1)
    iou = intersection / np.maximum(box_area[:, None] + truth_area[None, :] - intersection, 1e-9)
    candidates = np.flatnonzero((iou >= 0.5).any(axis=1))
    scores = np.asarray([item["score"] for item in predictions], dtype=np.float32)
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
    matched = np.zeros(len(truth), dtype=bool)
    for index in candidates:
        for target in np.argsort(-iou[index]):
            if iou[index, target] < 0.5:
                break
            if not matched[target]:
                labels[index] = 1
                matched[target] = True
                break
        if matched.all():
            break
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--ground-truth-source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--exclude-category", action="append", default=[])
    parser.add_argument(
        "--score-floor", type=float, default=0.0,
        help="Evaluate the historical score floor without rewriting the submission",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.score_floor <= 1:
        parser.error("--score-floor must be within [0,1]")

    allowed = [name for name in CLASS_NAMES if name not in set(args.exclude_category)]
    gt = load_ground_truth(args.ground_truth_source, args.split_manifest, args.split)
    class_scores = {name: array("f") for name in allowed}
    class_labels = {name: bytearray() for name in allowed}
    gt_counts: Counter[str] = Counter()
    seen_images = set()
    predictions_count = 0
    top_k_summary = {str(value): {"tp": 0, "predictions": 0} for value in TOP_K_VALUES}
    for image_id, grouped in groupby(iter_json_array(args.submission), key=lambda item: item["image_id"]):
        if image_id in seen_images:
            raise ValueError(f"Predictions for {image_id} are not contiguous")
        if image_id not in gt:
            raise ValueError(f"Unknown dev image: {image_id}")
        seen_images.add(image_id)
        by_class: dict[str, list[dict]] = defaultdict(list)
        for item in grouped:
            name = item["category_name"]
            if name not in allowed:
                raise ValueError(f"Unexpected category: {name}")
            if args.score_floor:
                item["score"] = max(float(item["score"]), args.score_floor)
            by_class[name].append(item)
            predictions_count += 1
        for name in allowed:
            truth = gt[image_id].get(name, [])
            gt_counts[name] += len(truth)
            items = by_class[name]
            labels = label_image_class(items, truth)
            class_scores[name].extend(float(item["score"]) for item in items)
            class_labels[name].extend(labels.tobytes())
            if items:
                scores = np.asarray([item["score"] for item in items], dtype=np.float32)
                ranked_labels = labels[np.argsort(-scores, kind="stable")]
                for value in TOP_K_VALUES:
                    selected = min(value, len(items))
                    top_k_summary[str(value)]["predictions"] += selected
                    top_k_summary[str(value)]["tp"] += int(ranked_labels[:selected].sum())

    if seen_images != set(gt):
        raise ValueError(f"Missing {len(set(gt) - seen_images)} dev images")
    per_class = {}
    threshold_summary = {str(value): {"tp": 0, "predictions": 0} for value in THRESHOLDS}
    for name in allowed:
        scores = np.frombuffer(class_scores[name], dtype=np.float32)
        labels = np.frombuffer(class_labels[name], dtype=np.uint8)
        per_class[name] = {
            "gt": gt_counts[name],
            "predictions": len(scores),
            "tp": int(labels.sum()),
            "ap50": ap50_from_ranked_labels(scores, labels, gt_counts[name]),
        }
        for threshold in THRESHOLDS:
            selected = scores >= threshold
            threshold_summary[str(threshold)]["tp"] += int(labels[selected].sum())
            threshold_summary[str(threshold)]["predictions"] += int(selected.sum())
    total_gt = sum(gt_counts.values())
    for item in threshold_summary.values():
        item["recall"] = item["tp"] / total_gt
        item["precision"] = item["tp"] / max(item["predictions"], 1)
    for item in top_k_summary.values():
        item["recall"] = item["tp"] / total_gt
        item["precision"] = item["tp"] / max(item["predictions"], 1)
    report = {
        "submission": str(args.submission.resolve()),
        "images": len(seen_images),
        "predictions": predictions_count,
        "ground_truth": total_gt,
        "score_floor": args.score_floor,
        "map50_macro": float(np.nanmean([item["ap50"] for item in per_class.values()])),
        "per_class": per_class,
        "thresholds": threshold_summary,
        "top_k_per_image_class": top_k_summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
