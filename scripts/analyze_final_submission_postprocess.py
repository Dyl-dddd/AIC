"""Calibrate post-processing that can be applied directly to a final submission.

The sweep is selected on both frozen development views. It never reads Test
labels and does not run a neural network.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import box_iou, classwise_nms, classwise_weighted_box_fusion
from steel_defect.inference import InferenceOptions, merge_candidates
from steel_defect.metrics import _match_class, average_precision, validate_predictions


PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
CLASS_IDS = {name: index for index, name in enumerate(PERMITTED)}
VIEW_WEIGHTS = {"original": 0.6192893401015228, "grid_crops": 0.38071065989847713}
FLOORS = (0.00003, 0.00005, 0.000075, 0.0001, 0.00015, 0.0002,
          0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.005, 0.01, 0.02,
          0.05, 0.1, 0.2, 0.4)
GEOMETRY_CONFIGS = (
    [("baseline", "none", 0.72)]
    + [(f"nms_{iou:.2f}", "nms", iou) for iou in (0.70, 0.68, 0.65, 0.62, 0.60, 0.55, 0.50, 0.45)]
    + [(f"vote_{iou:.2f}", "vote", iou) for iou in (0.70, 0.68, 0.65, 0.62, 0.60, 0.55)]
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cell(folder: Path) -> dict:
    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    with gzip.open(folder / "candidates.jsonl.gz", "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        rows = [json.loads(line) for line in handle]
    if header.get("signature") != report.get("signature"):
        raise ValueError(f"cache signature mismatch: {folder}")
    return {
        "rows": rows,
        "ground_truth": {row["image_id"]: row["ground_truth"] for row in rows},
        "sizes": {row["image_id"]: tuple(row["image_size"]) for row in rows},
        "signature": report["signature"],
    }


def replay(cell: dict) -> list[dict]:
    options = InferenceOptions(**cell["signature"]["options"])
    options = replace(options, conf=0.0001, iou=0.68, edge_penalty=0.50,
                      merge="nms", class_thresholds={})
    predictions = []
    for row in cell["rows"]:
        predictions.extend(
            {"image_id": row["image_id"], **item}
            for item in merge_candidates(
                row["candidates"], tuple(row["image_size"]), CLASS_NAMES, options
            )
            if item["category_name"] != "qilie"
        )
    return predictions


def fuse(model_a: list[dict], model_b: list[dict]) -> list[dict]:
    grouped: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for item in model_a:
        grouped[item["image_id"]].append((item, 1.0))
    for item in model_b:
        grouped[item["image_id"]].append((item, 0.35))
    output = []
    for image_id, rows in grouped.items():
        boxes = np.asarray([row[0]["bbox"] for row in rows], dtype=np.float32)
        scores = np.asarray(
            [min(1.0, float(row[0]["score"]) * row[1]) for row in rows],
            dtype=np.float32,
        )
        classes = np.asarray(
            [CLASS_IDS[row[0]["category_name"]] for row in rows], dtype=np.int64
        )
        for index in classwise_nms(boxes, scores, classes, 0.72):
            output.append({
                "image_id": image_id,
                "category_name": PERMITTED[int(classes[index])],
                "bbox": [int(round(value)) for value in boxes[index]],
                "score": round(float(scores[index]), 7),
            })
    return output


def _rows_by_image(predictions: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        grouped[item["image_id"]].append(item)
    return grouped


def _arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([item["bbox"] for item in rows], dtype=np.float32),
        np.asarray([item["score"] for item in rows], dtype=np.float32),
        np.asarray([CLASS_IDS[item["category_name"]] for item in rows], dtype=np.int64),
    )


def _serialize(image_id: str, boxes: np.ndarray, scores: np.ndarray,
               classes: np.ndarray) -> list[dict]:
    output = []
    for box, score, class_id in zip(boxes, scores, classes, strict=True):
        rounded = [int(round(float(value))) for value in box]
        if rounded[2] <= rounded[0] or rounded[3] <= rounded[1]:
            continue
        output.append({
            "image_id": image_id,
            "category_name": PERMITTED[int(class_id)],
            "bbox": rounded,
            "score": round(float(score), 7),
        })
    return output


def _vote_arrays(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                 iou: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    voted_boxes = []
    voted_scores = []
    voted_classes = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while len(order):
            anchor = int(order[0])
            overlaps = box_iou(boxes[anchor], boxes[order])
            cluster = order[overlaps > iou]
            weights = scores[cluster].astype(np.float64)
            voted = np.average(boxes[cluster].astype(np.float64), axis=0,
                               weights=np.maximum(weights, 1e-12))
            voted_boxes.append(voted.astype(np.float32))
            voted_scores.append(float(scores[anchor]))
            voted_classes.append(int(class_id))
            order = order[overlaps <= iou]
    order = np.argsort(np.asarray(voted_scores))[::-1]
    return (
        np.asarray(voted_boxes, dtype=np.float32)[order],
        np.asarray(voted_scores, dtype=np.float32)[order],
        np.asarray(voted_classes, dtype=np.int64)[order],
    )


def postprocess(predictions: list[dict], method: str, iou: float) -> list[dict]:
    if method == "none":
        return predictions
    output = []
    for image_id, rows in _rows_by_image(predictions).items():
        boxes, scores, classes = _arrays(rows)
        if method == "nms":
            keep = classwise_nms(boxes, scores, classes, iou)
            arrays = boxes[keep], scores[keep], classes[keep]
        elif method == "vote":
            arrays = _vote_arrays(boxes, scores, classes, iou)
        elif method == "wbf":
            arrays = classwise_weighted_box_fusion(boxes, scores, classes, iou)
        else:
            raise ValueError(f"unknown postprocess method: {method}")
        output.extend(_serialize(image_id, *arrays))
    return output


def metrics50(predictions: list[dict], gt: dict, sizes: dict) -> dict:
    validate_predictions(predictions, gt, PERMITTED, sizes)
    per_class = {}
    for name in PERMITTED:
        tp, fp, count = _match_class(predictions, gt, name, 0.5)
        per_class[name] = {
            "gt": count,
            "tp": int(tp.sum()),
            "fp": int(fp.sum()),
            "fn": count - int(tp.sum()),
            "ap50": average_precision(tp, fp, count),
        }
    ground_truth = sum(item["gt"] for item in per_class.values())
    tp = sum(item["tp"] for item in per_class.values())
    fp = sum(item["fp"] for item in per_class.values())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(ground_truth, 1)
    map50 = float(np.mean([item["ap50"] for item in per_class.values() if item["gt"]]))
    return {
        "p_micro": precision,
        "r_micro": recall,
        "map50": map50,
        "score": 20 * precision + 60 * recall + 20 * map50,
        "gt": ground_truth,
        "tp": tp,
        "fp": fp,
        "fn": ground_truth - tp,
        "predictions": len(predictions),
        "per_class": per_class,
    }


def summarize(records: dict[str, dict[str, dict]]) -> list[dict]:
    output = []
    for tag, views in records.items():
        if set(views) != set(VIEW_WEIGHTS):
            continue
        output.append({
            "tag": tag,
            "weighted_score": sum(VIEW_WEIGHTS[v] * views[v]["score"] for v in VIEW_WEIGHTS),
            "robust_min_score": min(views[v]["score"] for v in VIEW_WEIGHTS),
            "views": views,
        })
    output.sort(key=lambda item: (item["weighted_score"], item["robust_min_score"]), reverse=True)
    return output


def load_view(a_path: Path, b_path: Path) -> tuple[dict, list[dict]]:
    model_a = load_cell(a_path)
    model_b = load_cell(b_path)
    if model_a["ground_truth"] != model_b["ground_truth"] or model_a["sizes"] != model_b["sizes"]:
        raise ValueError("Model A/B development cells are not aligned")
    predictions = fuse(replay(model_a), replay(model_b))
    return model_a, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-original", type=Path, required=True)
    parser.add_argument("--model-a-grid", type=Path, required=True)
    parser.add_argument("--model-b-original", type=Path, required=True)
    parser.add_argument("--model-b-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        "original": (args.model_a_original, args.model_b_original),
        "grid_crops": (args.model_a_grid, args.model_b_grid),
    }

    geometry_records: dict[str, dict[str, dict]] = defaultdict(dict)
    for view, (a_path, b_path) in paths.items():
        cell, baseline = load_view(a_path, b_path)
        for tag, method, iou in GEOMETRY_CONFIGS:
            predictions = postprocess(baseline, method, iou)
            geometry_records[tag][view] = metrics50(
                predictions, cell["ground_truth"], cell["sizes"]
            )
            print(f"GEOMETRY {view} {tag}: {geometry_records[tag][view]['score']:.6f}", flush=True)
    geometry_grid = summarize(geometry_records)
    baseline_summary = next(item for item in geometry_grid if item["tag"] == "baseline")

    safe_geometry = [
        item for item in geometry_grid
        if all(item["views"][view]["score"] >= baseline_summary["views"][view]["score"]
               for view in VIEW_WEIGHTS)
    ][:6]
    if not safe_geometry:
        safe_geometry = [baseline_summary]
    safe_tags = {item["tag"] for item in safe_geometry} | {"baseline"}
    config_by_tag = {tag: (method, iou) for tag, method, iou in GEOMETRY_CONFIGS}

    threshold_records: dict[str, dict[str, dict]] = defaultdict(dict)
    for view, (a_path, b_path) in paths.items():
        cell, baseline = load_view(a_path, b_path)
        for geometry_tag in sorted(safe_tags):
            method, iou = config_by_tag[geometry_tag]
            geometry_predictions = postprocess(baseline, method, iou)
            for floor in FLOORS:
                tag = f"{geometry_tag}_floor_{floor:g}"
                selected = [item for item in geometry_predictions if float(item["score"]) >= floor]
                threshold_records[tag][view] = metrics50(
                    selected, cell["ground_truth"], cell["sizes"]
                )
                print(f"THRESHOLD {view} {tag}: {threshold_records[tag][view]['score']:.6f}", flush=True)
    threshold_grid = summarize(threshold_records)
    safe_thresholds = [
        item for item in threshold_grid
        if all(item["views"][view]["score"] >= baseline_summary["views"][view]["score"]
               for view in VIEW_WEIGHTS)
    ]
    decision = {
        "baseline": baseline_summary,
        "best_geometry": geometry_grid[0],
        "safe_geometry": safe_geometry,
        "best_threshold": threshold_grid[0],
        "safe_thresholds": safe_thresholds[:10],
        "recommended": (safe_thresholds[0] if safe_thresholds else geometry_grid[0]),
        "warning": "Frozen-development calibration only; official Test labels were not used.",
    }
    write_json(args.output / "geometry_grid.json", geometry_grid)
    write_json(args.output / "threshold_grid.json", threshold_grid)
    write_json(args.output / "decision.json", decision)

    lines = [
        "# Direct-submission post-processing sweep",
        "",
        "All scores are frozen-development proxies, not official Test results.",
        "",
        "## Geometry-only top results",
        "",
        "|Tag|Original|Grid|Weighted|Worst|Predictions original/grid|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in geometry_grid[:12]:
        lines.append(
            f"|{item['tag']}|{item['views']['original']['score']:.3f}|"
            f"{item['views']['grid_crops']['score']:.3f}|{item['weighted_score']:.3f}|"
            f"{item['robust_min_score']:.3f}|{item['views']['original']['predictions']}/"
            f"{item['views']['grid_crops']['predictions']}|"
        )
    lines.extend([
        "", "## Geometry plus global-floor top results", "",
        "|Tag|Original|Grid|Weighted|Worst|TP original/grid|FP original/grid|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for item in threshold_grid[:15]:
        lines.append(
            f"|{item['tag']}|{item['views']['original']['score']:.3f}|"
            f"{item['views']['grid_crops']['score']:.3f}|{item['weighted_score']:.3f}|"
            f"{item['robust_min_score']:.3f}|{item['views']['original']['tp']}/"
            f"{item['views']['grid_crops']['tp']}|{item['views']['original']['fp']}/"
            f"{item['views']['grid_crops']['fp']}|"
        )
    lines.extend(["", f"Recommended: `{decision['recommended']['tag']}`."])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

