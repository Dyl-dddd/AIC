"""Measure which frozen V7 misses are recoverable by V9 P1/P2 predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from steel_defect.geometry import box_iou


def group(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    output: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        output[(row["image_id"], row["category_name"])].append(row)
    return output


def overlaps(box: list[float], rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float32)
    return box_iou(
        np.asarray(box, dtype=np.float32),
        np.asarray([row["bbox"] for row in rows], dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--v9-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"status": "analysis_only", "views": {}}
    for view in ("original", "grid_crops"):
        print(f"LOAD {view}", flush=True)
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        p1 = json.loads((args.v9_cells / f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}" / "predictions.json").read_text())
        p2 = json.loads((args.v9_cells / f"v9_model_b_quality_p2_last_224153cc09_{view}" / "predictions.json").read_text())
        groups = {"v7": group(baseline), "p1": group(p1), "p2": group(p2)}
        records = []
        for image_id, classes in cell_a["ground_truth"].items():
            for class_name, boxes in classes.items():
                key = image_id, class_name
                for gt_index, gt_box in enumerate(boxes):
                    row = {"image_id": image_id, "class_name": class_name, "gt_index": gt_index}
                    best_rows = {}
                    for model in ("v7", "p1", "p2"):
                        model_rows = groups[model].get(key, [])
                        values = overlaps(gt_box, model_rows)
                        if len(values):
                            index = int(np.argmax(values))
                            best = model_rows[index]
                            best_rows[model] = best
                            row[f"{model}_iou"] = float(values[index])
                            row[f"{model}_score"] = float(best["score"])
                        else:
                            row[f"{model}_iou"] = 0.0
                            row[f"{model}_score"] = 0.0
                    p1_best = best_rows.get("p1")
                    if p1_best:
                        support = overlaps(p1_best["bbox"], groups["p2"].get(key, []))
                        base = overlaps(p1_best["bbox"], groups["v7"].get(key, []))
                        row["p1_p2_support_iou"] = float(np.max(support)) if len(support) else 0.0
                        row["p1_base_iou"] = float(np.max(base)) if len(base) else 0.0
                    records.append(row)
        missed = [row for row in records if row["v7_iou"] < 0.50]
        recoverable_p1 = [row for row in missed if row["p1_iou"] >= 0.50]
        recoverable_p2 = [row for row in missed if row["p2_iou"] >= 0.50]
        recoverable_both = [row for row in missed if row["p1_iou"] >= 0.50 and row["p2_iou"] >= 0.50]
        baseline_metrics = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        payload["views"][view] = {
            "baseline": baseline_metrics,
            "coverage_counts": {
                "gt": len(records),
                "v7_missed_by_independent_coverage": len(missed),
                "v7_misses_covered_by_p1": len(recoverable_p1),
                "v7_misses_covered_by_p2": len(recoverable_p2),
                "v7_misses_covered_by_both": len(recoverable_both),
            },
            "recoverable_p1": recoverable_p1,
        }
        print(json.dumps(payload["views"][view]["coverage_counts"]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
