"""Test one preregistered cross-view quota rescue rule against the exact V7 baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_final_submission_postprocess import load_cell
from scripts.analyze_quality_rescue import summarize


SUPPORT_IOU = 0.60
NOVELTY_IOU = 0.30
PER_SOURCE_QUOTA = 1
TILE_PATTERN = re.compile(r"^(?P<source>.+)::x(?P<x>\d+)_y(?P<y>\d+)$")


def source_and_global_box(prediction: dict) -> tuple[str, np.ndarray]:
    match = TILE_PATTERN.fullmatch(prediction["image_id"])
    box = np.asarray(prediction["bbox"], dtype=np.float32)
    if not match:
        return prediction["image_id"], box
    offset = np.asarray([
        float(match.group("x")), float(match.group("y")),
        float(match.group("x")), float(match.group("y")),
    ], dtype=np.float32)
    return match.group("source"), box + offset


def iou_one_to_many(box: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
    if not others:
        return np.zeros(0, dtype=np.float32)
    array = np.asarray(others, dtype=np.float32)
    upper_left = np.maximum(box[:2], array[:, :2])
    lower_right = np.minimum(box[2:], array[:, 2:])
    intersection_side = np.maximum(lower_right - upper_left, 0.0)
    intersection = intersection_side[:, 0] * intersection_side[:, 1]
    area = max(float((box[2] - box[0]) * (box[3] - box[1])), 0.0)
    other_area = np.maximum(array[:, 2] - array[:, 0], 0.0) * np.maximum(
        array[:, 3] - array[:, 1], 0.0
    )
    return intersection / np.maximum(area + other_area - intersection, 1e-12)


def grouped_global(predictions: list[dict]) -> dict[tuple[str, str], list[np.ndarray]]:
    result: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for item in predictions:
        source, box = source_and_global_box(item)
        result[(source, item["category_name"])].append(box)
    return result


def select_pairs(
    quality_original: list[dict], quality_grid: list[dict],
    v7_original: list[dict], v7_grid: list[dict],
) -> list[dict]:
    grid_quality: dict[tuple[str, str], list[tuple[dict, np.ndarray]]] = defaultdict(list)
    for item in quality_grid:
        source, box = source_and_global_box(item)
        grid_quality[(source, item["category_name"])].append((item, box))
    v7_original_groups = grouped_global(v7_original)
    v7_grid_groups = grouped_global(v7_grid)
    candidates: dict[str, list[dict]] = defaultdict(list)
    for original in quality_original:
        source, original_box = source_and_global_box(original)
        key = (source, original["category_name"])
        grid_items = grid_quality.get(key, [])
        if not grid_items:
            continue
        overlaps = iou_one_to_many(original_box, [item[1] for item in grid_items])
        best = int(np.argmax(overlaps))
        support = float(overlaps[best])
        if support < SUPPORT_IOU:
            continue
        grid_item, grid_box = grid_items[best]
        original_existing = iou_one_to_many(original_box, v7_original_groups.get(key, []))
        grid_existing = iou_one_to_many(grid_box, v7_grid_groups.get(key, []))
        if (len(original_existing) and float(original_existing.max()) >= NOVELTY_IOU) or (
            len(grid_existing) and float(grid_existing.max()) >= NOVELTY_IOU
        ):
            continue
        rank = support * np.sqrt(
            max(float(original["score"]), 1e-12) * max(float(grid_item["score"]), 1e-12)
        )
        candidates[source].append({
            "source": source,
            "category_name": original["category_name"],
            "original": original,
            "grid": grid_item,
            "support_iou": support,
            "rank": float(rank),
        })
    selected = []
    for source in sorted(candidates):
        selected.extend(sorted(
            candidates[source], key=lambda item: (item["rank"], item["support_iou"]),
            reverse=True,
        )[:PER_SOURCE_QUOTA])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = {}
    baseline = {}
    for view in ("original", "grid_crops"):
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        cells[view] = cell_a
        baseline[view] = fuse_v7(cell_a, cell_b)
    quality_original = json.loads((
        args.quality_evaluation / "varifocal/varifocal_original_predictions.json"
    ).read_text())
    quality_grid = json.loads((
        args.quality_evaluation / "varifocal/varifocal_grid_crops_predictions.json"
    ).read_text())
    selected = select_pairs(
        quality_original, quality_grid, baseline["original"], baseline["grid_crops"]
    )
    result = {
        "rule": {
            "support_iou": SUPPORT_IOU,
            "novelty_iou": NOVELTY_IOU,
            "per_source_quota": PER_SOURCE_QUOTA,
            "ranking": "cross_view_iou * geometric_mean(original_score, grid_score)",
            "rescue_score": "below minimum V7 score; membership-only tail rescue",
        },
        "selected_count": len(selected),
        "views": {},
    }
    for view in ("original", "grid_crops"):
        tail_score = min(float(item["score"]) for item in baseline[view]) * 0.5
        rescue = []
        for item in selected:
            source_item = item["original" if view == "original" else "grid"]
            rescue.append({
                "image_id": source_item["image_id"],
                "category_name": source_item["category_name"],
                "bbox": source_item["bbox"],
                "score": tail_score * (0.5 + 0.5 * item["rank"]),
            })
        base_metrics = summarize(
            baseline[view], cells[view]["ground_truth"], cells[view]["sizes"]
        )
        rescue_metrics = summarize(
            [*baseline[view], *rescue], cells[view]["ground_truth"], cells[view]["sizes"]
        )
        result["views"][view] = {
            "baseline": base_metrics,
            "rescue": rescue_metrics,
            "delta": {
                key: float(rescue_metrics[key]) - float(base_metrics[key])
                for key in (
                    "score_proxy_20_60_20", "map50_macro", "map75_macro",
                    "map50_95_macro", "precision_micro", "recall_micro",
                    "false_positives_per_image",
                )
            },
        }
    result["selected"] = [{
        "source": item["source"],
        "category_name": item["category_name"],
        "support_iou": item["support_iou"],
        "rank": item["rank"],
        "original_image_id": item["original"]["image_id"],
        "grid_image_id": item["grid"]["image_id"],
    } for item in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_count": len(selected),
        "views": {view: result["views"][view]["delta"] for view in result["views"]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
