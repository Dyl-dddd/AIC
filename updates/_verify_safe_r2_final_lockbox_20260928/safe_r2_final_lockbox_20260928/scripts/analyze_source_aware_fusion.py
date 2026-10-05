"""Evaluate source-aware Model A/B consensus ranking on frozen development caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import (
    CLASS_IDS,
    PERMITTED,
    VIEW_WEIGHTS,
    load_cell,
    metrics50,
    replay,
    summarize,
    write_json,
)
from steel_defect.geometry import box_iou, classwise_nms


MATCH_IOUS = (0.50, 0.60, 0.70)
BOOSTS = (1.10, 1.25, 1.50, 2.00)
COORD_MODES = ("anchor", "vote")
UNMATCHED_B_FACTORS = (0.50, 1.00)


def source_aware_fuse(
    model_a: list[dict],
    model_b: list[dict],
    match_iou: float,
    boost: float,
    coord_mode: str,
    unmatched_b_factor: float,
    b_scale: float = 0.35,
    final_nms_iou: float = 0.72,
) -> list[dict]:
    grouped_a: dict[tuple[str, str], list[dict]] = defaultdict(list)
    grouped_b: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in model_a:
        grouped_a[(item["image_id"], item["category_name"])].append(item)
    for item in model_b:
        grouped_b[(item["image_id"], item["category_name"])].append(item)

    by_image: dict[str, list[dict]] = defaultdict(list)
    for image_id, class_name in sorted(set(grouped_a) | set(grouped_b)):
        rows_a = sorted(grouped_a[(image_id, class_name)], key=lambda x: float(x["score"]), reverse=True)
        rows_b = grouped_b[(image_id, class_name)]
        available_b = set(range(len(rows_b)))
        for item_a in rows_a:
            box_a = np.asarray(item_a["bbox"], dtype=np.float32)
            best = None
            if available_b:
                indices = sorted(available_b)
                boxes_b = np.asarray([rows_b[index]["bbox"] for index in indices], dtype=np.float32)
                ious = box_iou(box_a, boxes_b)
                local = int(np.argmax(ious))
                if float(ious[local]) >= match_iou:
                    best = indices[local]
            score_a = float(item_a["score"])
            if best is None:
                box = box_a
                score = score_a
                consensus = False
            else:
                available_b.remove(best)
                item_b = rows_b[best]
                box_b = np.asarray(item_b["bbox"], dtype=np.float32)
                score_b = float(item_b["score"]) * b_scale
                if coord_mode == "anchor":
                    box = box_a if score_a >= score_b else box_b
                elif coord_mode == "vote":
                    weights = np.asarray([max(score_a, 1e-12), max(score_b, 1e-12)])
                    box = np.average(np.stack([box_a, box_b]), axis=0, weights=weights)
                else:
                    raise ValueError(f"unknown coordinate mode: {coord_mode}")
                score = min(1.0, max(score_a, score_b) * boost)
                consensus = True
            by_image[image_id].append({
                "category_name": class_name,
                "bbox": box,
                "score": score,
                "consensus": consensus,
            })
        for index in sorted(available_b):
            item_b = rows_b[index]
            by_image[image_id].append({
                "category_name": class_name,
                "bbox": np.asarray(item_b["bbox"], dtype=np.float32),
                "score": float(item_b["score"]) * b_scale * unmatched_b_factor,
                "consensus": False,
            })

    output = []
    for image_id, rows in by_image.items():
        boxes = np.asarray([item["bbox"] for item in rows], dtype=np.float32)
        scores = np.asarray([item["score"] for item in rows], dtype=np.float32)
        classes = np.asarray([CLASS_IDS[item["category_name"]] for item in rows], dtype=np.int64)
        for index in classwise_nms(boxes, scores, classes, final_nms_iou):
            box = [int(round(float(value))) for value in boxes[index]]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            output.append({
                "image_id": image_id,
                "category_name": PERMITTED[int(classes[index])],
                "bbox": box,
                "score": round(float(scores[index]), 7),
            })
    return output


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
    records: dict[str, dict[str, dict]] = defaultdict(dict)
    for view, (a_path, b_path) in paths.items():
        cell_a = load_cell(a_path)
        cell_b = load_cell(b_path)
        if cell_a["ground_truth"] != cell_b["ground_truth"] or cell_a["sizes"] != cell_b["sizes"]:
            raise ValueError("unaligned Model A/B cells")
        model_a = replay(cell_a)
        model_b = replay(cell_b)
        for match_iou in MATCH_IOUS:
            for boost in BOOSTS:
                for coord_mode in COORD_MODES:
                    for unmatched_b_factor in UNMATCHED_B_FACTORS:
                        tag = (
                            f"consensus_iou{match_iou:.2f}_boost{boost:.2f}_"
                            f"{coord_mode}_ub{unmatched_b_factor:.2f}"
                        )
                        predictions = source_aware_fuse(
                            model_a, model_b, match_iou, boost, coord_mode,
                            unmatched_b_factor,
                        )
                        records[tag][view] = metrics50(
                            predictions, cell_a["ground_truth"], cell_a["sizes"]
                        )
                        print(f"{view} {tag}: {records[tag][view]['score']:.6f}", flush=True)
    grid = summarize(records)
    robust = [
        item for item in grid
        if item["views"]["original"]["score"] >= 68.34617726527127
        and item["views"]["grid_crops"]["score"] >= 68.21574899659251
    ]
    decision = {
        "best": grid[0],
        "robust_candidates": robust[:10],
        "recommended": robust[0] if robust else None,
        "baseline_weighted": 68.29652183303315,
        "warning": "Frozen-development source-aware calibration; no Test labels used.",
    }
    write_json(args.output / "grid.json", grid)
    write_json(args.output / "decision.json", decision)
    lines = [
        "# Source-aware consensus fusion", "",
        "All values are frozen-development proxies.", "",
        "|Tag|Original|Grid|Weighted|Worst|TP original/grid|FP original/grid|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in grid[:20]:
        lines.append(
            f"|{item['tag']}|{item['views']['original']['score']:.3f}|"
            f"{item['views']['grid_crops']['score']:.3f}|{item['weighted_score']:.3f}|"
            f"{item['robust_min_score']:.3f}|{item['views']['original']['tp']}/"
            f"{item['views']['grid_crops']['tp']}|{item['views']['original']['fp']}/"
            f"{item['views']['grid_crops']['fp']}|"
        )
    lines.extend(["", f"Robust candidate count: `{len(robust)}`."])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

