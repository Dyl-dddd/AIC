"""Frozen-dev comparison of NMS, WBF, and seam-aware WBF on cached candidates.

This script never runs a model, reads Test labels, or writes a submission.
The primary seam policy matches the referenced fabric-defect approach: drop
tile candidates touching an internal seam before WBF.  The optional orphan
policy keeps seam candidates when no intact same-class candidate covers them.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import box_iou
from steel_defect.inference import InferenceOptions, inference_tiles, merge_candidates


DEFAULT_CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"


def count_flags(rows: list[dict], options: InferenceOptions) -> dict:
    counts: Counter = Counter()
    for row in rows:
        counts["images"] += 1
        width, height = row["image_size"]
        tiles = inference_tiles(width, height, options)
        vertical_lines = {
            value for tile in tiles for value in (tile.x, tile.x + tile.width)
            if 0 < value < width
        }
        horizontal_lines = {
            value for tile in tiles for value in (tile.y, tile.y + tile.height)
            if 0 < value < height
        }
        for candidate in row["candidates"]:
            counts["total"] += 1
            if candidate.get("is_global"):
                counts["global"] += 1
            if candidate.get("touches_internal_edge"):
                counts["internal_seam"] += 1
                class_name = CLASS_NAMES[int(candidate["class_id"])]
                counts[f"seam_{class_name}"] += 1
                x1, y1, x2, y2 = candidate["bbox"]
                vertical = any(min(abs(x1 - x), abs(x2 - x)) <= options.edge_margin + 0.1
                               for x in vertical_lines)
                horizontal = any(min(abs(y1 - y), abs(y2 - y)) <= options.edge_margin + 0.1
                                 for y in horizontal_lines)
                if vertical:
                    counts["vertical_seam"] += 1
                    counts[f"vertical_seam_{class_name}"] += 1
                if horizontal:
                    counts["horizontal_seam"] += 1
                    counts[f"horizontal_seam_{class_name}"] += 1
                if vertical and horizontal:
                    counts["both_seams"] += 1
                if not vertical and not horizontal:
                    counts["seam_direction_unclassified"] += 1
    return dict(counts)


def keep_orphans(candidates: list[dict], threshold: float) -> list[dict]:
    """Exclude seam geometry only when a complete box supports that defect.

    IoU is supplemented by intersection / seam-box area because a truncated
    stump may have low IoU with its correct full-width/full-height counterpart.
    The single threshold is the same as WBF's threshold, not a fitted parameter.
    """
    complete_by_class: dict[int, list[dict]] = {}
    for item in candidates:
        if not item.get("touches_internal_edge"):
            complete_by_class.setdefault(int(item["class_id"]), []).append(item)
    kept = []
    for item in candidates:
        if not item.get("touches_internal_edge"):
            kept.append(item)
            continue
        complete = complete_by_class.get(int(item["class_id"]), [])
        if not complete:
            kept.append(item)
            continue
        box = np.asarray(item["bbox"], dtype=np.float32)
        boxes = np.asarray([other["bbox"] for other in complete], dtype=np.float32)
        ious = box_iou(box, boxes)
        intersection = np.maximum(0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])) * np.maximum(0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))
        box_area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-9)
        if not np.any((ious > threshold) | (intersection / box_area > threshold)):
            kept.append(item)
    return kept


def evaluate_cell(cell: dict, methods: list[str]) -> dict:
    options = InferenceOptions(**cell["signature"]["options"])
    options = replace(options, conf=0.0001, iou=0.68, edge_penalty=0.50,
                      class_thresholds={})
    results = {}
    for method in methods:
        merge = "nms" if method in {"nms", "seam_drop_nms"} else "wbf"
        method_options = replace(options, merge=merge)
        predictions = []
        retained_candidates = 0
        for row in cell["rows"]:
            candidates = row["candidates"]
            if method in {"seam_drop_wbf", "seam_drop_nms"}:
                candidates = [c for c in candidates if not c.get("touches_internal_edge")]
            elif method == "seam_orphan_wbf":
                candidates = keep_orphans(candidates, options.iou)
            retained_candidates += len(candidates)
            predictions.extend(
                {"image_id": row["image_id"], **item}
                for item in merge_candidates(
                    candidates, tuple(row["image_size"]), CLASS_NAMES, method_options
                )
                if item["category_name"] != "qilie"
            )
        result = metrics50(predictions, cell["ground_truth"], cell["sizes"])
        result["retained_candidates"] = retained_candidates
        results[method] = result
        print(f"{method}: score={result['score']:.5f} AP50={result['map50']:.5f} "
              f"P={result['p_micro']:.5f} R={result['r_micro']:.5f} "
              f"TP={result['tp']} FP={result['fp']} FN={result['fn']}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--methods", nargs="+", default=["nms", "wbf", "seam_drop_wbf"],
                        choices=("nms", "seam_drop_nms", "wbf", "seam_drop_wbf", "seam_orphan_wbf"))
    parser.add_argument("--limit", type=int, default=0, help="Diagnostic subset only; never use for a deployment decision")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report: dict = {
        "purpose": "frozen-development seam-aware fusion ablation",
        "warning": "Cached V6/V7 single-model development proxy, not the V12 67.92 official baseline.",
        "fixed_parameters": {"conf": 0.0001, "iou": 0.68, "edge_penalty": 0.50},
        "models": {},
    }
    for model in args.models:
        report["models"][model] = {}
        for view in ("original", "grid_crops"):
            folder = args.cells / f"model_{model}_tta_s{1024 if model == 'a' else 1280}_{view}"
            print(f"LOAD {folder}", flush=True)
            cell = load_cell(folder)
            if args.limit:
                cell["rows"] = cell["rows"][:args.limit]
                selected = {row["image_id"] for row in cell["rows"]}
                cell["ground_truth"] = {k: v for k, v in cell["ground_truth"].items() if k in selected}
                cell["sizes"] = {k: v for k, v in cell["sizes"].items() if k in selected}
            stats = count_flags(cell["rows"], InferenceOptions(**cell["signature"]["options"]))
            print(f"{model}/{view}: {stats}", flush=True)
            report["models"][model][view] = {
                "candidate_flags": stats,
                "cache_signature": cell["signature"],
                "metrics": evaluate_cell(cell, args.methods),
            }
    report["limit"] = args.limit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
