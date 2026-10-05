"""VOC-style detection metrics computed from dataset-provided ground truth."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .geometry import box_iou
from .voc import VocRecord


def validate_predictions(predictions, ground_truth, class_names, image_sizes=None):
    if len(class_names) != len(set(class_names)):
        raise ValueError("Duplicate class names")
    names = set(class_names)
    for image_id, per_class in ground_truth.items():
        if set(per_class) - names:
            raise ValueError(f"Unknown ground-truth class in {image_id}")
        for boxes in per_class.values():
            for box in boxes:
                _validate_box(box, image_id, image_sizes)
    for pred in predictions:
        if pred["image_id"] not in ground_truth:
            raise ValueError(f"Unknown prediction image_id: {pred['image_id']}")
        if pred["category_name"] not in names:
            raise ValueError(f"Unknown prediction class: {pred['category_name']}")
        score = float(pred["score"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Prediction score must be finite and within [0,1]")
        _validate_box(pred["bbox"], pred["image_id"], image_sizes)


def _validate_box(box, image_id, image_sizes):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Box must contain four finite coordinates")
    x1, y1, x2, y2 = values
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid box geometry")
    if image_sizes is not None:
        width, height = image_sizes[image_id]
        if x2 > width or y2 > height:
            raise ValueError("Box outside image bounds")


def ground_truth_from_records(records: Iterable[VocRecord], image_ids: dict[str, str]) -> dict[str, dict[str, list[list[float]]]]:
    ground_truth: dict[str, dict[str, list[list[float]]]] = {}
    for record in records:
        image_id = image_ids[str(record.image_path.resolve())]
        per_class: dict[str, list[list[float]]] = defaultdict(list)
        for annotation in record.annotations:
            per_class[annotation.class_name].append(list(annotation.box))
        ground_truth[image_id] = dict(per_class)
    return ground_truth


def _match_class(
    predictions: list[dict],
    ground_truth: dict[str, dict[str, list[list[float]]]],
    class_name: str,
    iou_threshold: float,
    score_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    selected = sorted(
        (
            prediction for prediction in predictions
            if prediction["category_name"] == class_name and float(prediction["score"]) >= score_threshold
        ),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    total_gt = sum(len(per_class.get(class_name, [])) for per_class in ground_truth.values())
    matched: dict[str, set[int]] = defaultdict(set)
    tp = np.zeros(len(selected), dtype=np.float64)
    fp = np.zeros(len(selected), dtype=np.float64)
    for index, prediction in enumerate(selected):
        image_id = prediction["image_id"]
        boxes = np.asarray(ground_truth.get(image_id, {}).get(class_name, []), dtype=np.float32)
        if not len(boxes):
            fp[index] = 1.0
            continue
        ious = box_iou(np.asarray(prediction["bbox"], dtype=np.float32), boxes)
        order = np.argsort(ious)[::-1]
        match = next((int(candidate) for candidate in order if candidate not in matched[image_id]), None)
        if match is not None and float(ious[match]) >= iou_threshold:
            tp[index] = 1.0
            matched[image_id].add(match)
        else:
            fp[index] = 1.0
    return tp, fp, total_gt


def average_precision(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt <= 0:
        return float("nan")
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / total_gt
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    interpolated = []
    for level in np.linspace(0, 1, 101):
        values = precision[recall >= level]
        interpolated.append(float(values.max()) if values.size else 0.0)
    return float(np.mean(interpolated))


def operating_metrics(tp: np.ndarray, fp: np.ndarray, total_gt: int, beta: float = 2.0) -> dict[str, float]:
    true_positive = float(tp.sum())
    false_positive = float(fp.sum())
    precision = true_positive / max(true_positive + false_positive, 1e-12)
    recall = true_positive / max(total_gt, 1)
    beta2 = beta * beta
    f_beta = (1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, f"f{beta:g}": f_beta}


def evaluate_predictions(
    predictions: list[dict],
    ground_truth: dict[str, dict[str, list[list[float]]]],
    class_names: list[str],
    operating_threshold: float = 0.05,
    beta: float = 2.0,
    class_thresholds: dict[str, float] | None = None,
    image_sizes: dict[str, tuple[int, int]] | None = None,
) -> dict:
    class_thresholds = class_thresholds or {}
    validate_predictions(predictions, ground_truth, class_names, image_sizes)
    if set(class_thresholds) - set(class_names):
        raise ValueError("Unknown threshold class")
    if any(not np.isfinite(value) or not 0 <= value <= 1
           for value in [operating_threshold, *class_thresholds.values()]):
        raise ValueError("Thresholds must be finite and within [0,1]")
    iou_thresholds = np.arange(0.50, 0.96, 0.05)
    per_class = {}
    for class_name in class_names:
        aps = []
        ap50 = float("nan")
        ap75 = float("nan")
        for iou_threshold in iou_thresholds:
            tp, fp, total_gt = _match_class(predictions, ground_truth, class_name, float(iou_threshold))
            ap = average_precision(tp, fp, total_gt)
            aps.append(ap)
            if np.isclose(iou_threshold, 0.50):
                ap50 = ap
            if np.isclose(iou_threshold, 0.75):
                ap75 = ap
        class_operating_threshold = class_thresholds.get(class_name, operating_threshold)
        tp, fp, total_gt = _match_class(
            predictions, ground_truth, class_name, 0.50, score_threshold=class_operating_threshold
        )
        per_class[class_name] = {
            "ground_truth": total_gt,
            "predictions": int(len(tp)),
            "true_positives": int(tp.sum()),
            "false_positives": int(fp.sum()),
            "operating_threshold": class_operating_threshold,
            "ap50": ap50,
            "ap75": ap75,
            "map50_95": float(np.nanmean(aps)) if not np.all(np.isnan(aps)) else float("nan"),
            **operating_metrics(tp, fp, total_gt, beta=beta),
        }
    valid = [metrics for metrics in per_class.values() if metrics["ground_truth"] > 0]
    summary = {
        "map50_macro": float(np.mean([item["ap50"] for item in valid])) if valid else float("nan"),
        "map75_macro": float(np.mean([item["ap75"] for item in valid])) if valid else float("nan"),
        "map50_95_macro": float(np.mean([item["map50_95"] for item in valid])) if valid else float("nan"),
        "precision_macro": float(np.mean([item["precision"] for item in valid])) if valid else float("nan"),
        "recall_macro": float(np.mean([item["recall"] for item in valid])) if valid else float("nan"),
        f"f{beta:g}_macro": float(np.mean([item[f"f{beta:g}"] for item in valid])) if valid else float("nan"),
        "operating_threshold": operating_threshold,
        "class_thresholds": class_thresholds,
        "false_positives_per_image": (
            float(sum(item["false_positives"] for item in per_class.values())) / max(len(ground_truth), 1)
        ),
    }
    tail = [per_class[name] for name in ("qilie", "huashang") if name in per_class and per_class[name]["ground_truth"] > 0]
    if tail:
        summary["tail_map50_macro"] = float(np.mean([item["ap50"] for item in tail]))
        summary["tail_recall_macro"] = float(np.mean([item["recall"] for item in tail]))
    return {"summary": summary, "per_class": per_class}


def tune_class_thresholds(
    predictions: list[dict],
    ground_truth: dict[str, dict[str, list[list[float]]]],
    class_names: list[str],
    beta: float = 2.0,
    minimum: float = 0.01,
    maximum: float = 0.50,
    steps: int = 50,
) -> dict[str, float]:
    thresholds = np.linspace(minimum, maximum, steps)
    selected: dict[str, float] = {}
    for class_name in class_names:
        best = (-1.0, 0.0, minimum)
        for threshold in thresholds:
            tp, fp, total_gt = _match_class(
                predictions, ground_truth, class_name, 0.50, score_threshold=float(threshold)
            )
            metrics = operating_metrics(tp, fp, total_gt, beta=beta)
            candidate = (metrics[f"f{beta:g}"], metrics["recall"], -float(threshold))
            if candidate > best:
                best = candidate
                selected[class_name] = float(threshold)
    return selected
