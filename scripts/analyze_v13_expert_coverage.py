"""Audit V13 expert coverage against the protected V7/V12 candidate universe."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from steel_defect.geometry import box_iou


def grouped(predictions: list[dict]) -> dict[tuple[str, str], list[dict]]:
    output: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in predictions:
        output[(item["image_id"], item["category_name"])].append(item)
    for rows in output.values():
        rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return output


def coverage(predictions: list[dict], ground_truth: dict) -> tuple[set[tuple], dict[tuple, dict]]:
    by_key = grouped(predictions)
    hits: set[tuple] = set()
    details: dict[tuple, dict] = {}
    for image_id, per_class in ground_truth.items():
        for class_name, gt_boxes in per_class.items():
            rows = by_key.get((image_id, class_name), [])
            boxes = np.asarray([item["bbox"] for item in rows], dtype=np.float32)
            scores = np.asarray([item["score"] for item in rows], dtype=np.float64)
            for gt_index, gt_box in enumerate(gt_boxes):
                key = (image_id, class_name, gt_index)
                if not len(rows):
                    details[key] = {"hit": False, "max_iou": 0.0, "best_score": 0.0, "rank": None}
                    continue
                ious = box_iou(np.asarray(gt_box, dtype=np.float32), boxes)
                matching = np.flatnonzero(ious >= 0.50)
                if not len(matching):
                    details[key] = {
                        "hit": False,
                        "max_iou": float(np.max(ious)),
                        "best_score": 0.0,
                        "rank": None,
                    }
                    continue
                best = int(matching[np.argmax(scores[matching])])
                hits.add(key)
                details[key] = {
                    "hit": True,
                    "max_iou": float(np.max(ious)),
                    "best_score": float(scores[best]),
                    "rank": best + 1,
                }
    return hits, details


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p75": ordered[int(round(0.75 * (len(ordered) - 1)))],
        "p90": ordered[int(round(0.90 * (len(ordered) - 1)))],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--v13-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"views": {}}
    for view in ("original", "grid_crops"):
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        gt = cell_a["ground_truth"]
        sizes = cell_a["sizes"]
        baseline_hits, baseline_details = coverage(baseline, gt)
        view_result = {
            "baseline": metrics50(baseline, gt, sizes),
            "experts": {},
        }
        folders = {
            "localize_vfl": next(args.v13_cells.glob(f"v13_hires_localize_vfl_*_{view}")),
            "recall_bce": next(args.v13_cells.glob(f"v13_hires_recall_bce_*_{view}")),
        }
        for name, folder in folders.items():
            predictions = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
            expert_hits, details = coverage(predictions, gt)
            unique = expert_hits - baseline_hits
            lost = baseline_hits - expert_hits
            by_class = Counter(key[1] for key in unique)
            unique_scores = [details[key]["best_score"] for key in unique]
            unique_ranks = [details[key]["rank"] for key in unique]
            baseline_misses = set(details) - baseline_hits
            near_miss_ious = [details[key]["max_iou"] for key in baseline_misses if not details[key]["hit"]]
            view_result["experts"][name] = {
                "metrics": metrics50(predictions, gt, sizes),
                "coverage_hits": len(expert_hits),
                "unique_vs_v7": len(unique),
                "v7_only": len(lost),
                "union_hits": len(expert_hits | baseline_hits),
                "union_recall": len(expert_hits | baseline_hits) / max(len(details), 1),
                "unique_by_class": dict(sorted(by_class.items())),
                "unique_hit_score_distribution": distribution(unique_scores),
                "unique_hit_rank_distribution": distribution(unique_ranks),
                "expert_miss_max_iou_distribution": distribution(near_miss_ious),
                "unique_instances": [
                    {"image_id": key[0], "class_name": key[1], "gt_index": key[2], **details[key]}
                    for key in sorted(unique)
                ],
            }
        result["views"][view] = view_result

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# V13 expert coverage audit", "",
        "Coverage means at least one same-class box reaches IoU >= 0.50 for a frozen GT instance.", "",
        "|view|expert|expert TP|unique vs V7|V7-only|union hits|union recall|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for view, values in result["views"].items():
        for name, expert in values["experts"].items():
            lines.append(
                f"|{view}|{name}|{expert['coverage_hits']}|{expert['unique_vs_v7']}|"
                f"{expert['v7_only']}|{expert['union_hits']}|{expert['union_recall']:.6f}|"
            )
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
