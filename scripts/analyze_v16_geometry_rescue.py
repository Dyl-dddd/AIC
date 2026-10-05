"""Evaluate V13 localization box transfer on frozen V7 development predictions."""

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


def match_expert(baseline: list[dict], expert: list[dict]) -> list[dict | None]:
    groups = defaultdict(list)
    for row in expert:
        groups[(row["image_id"], row["category_name"])].append(row)
    arrays = {key: np.asarray([item["bbox"] for item in rows], dtype=np.float32) for key, rows in groups.items()}
    output = []
    for index, row in enumerate(baseline):
        pool = groups.get((row["image_id"], row["category_name"]), [])
        if not pool:
            output.append(None)
            continue
        boxes = arrays[(row["image_id"], row["category_name"])]
        overlaps = box_iou(np.asarray(row["bbox"], dtype=np.float32), boxes)
        best = int(np.argmax(overlaps))
        output.append({"iou": float(overlaps[best]), "box": pool[best]["bbox"], "score": float(pool[best]["score"])})
        if index % 50000 == 0:
            print(f"MATCH {index}/{len(baseline)}", flush=True)
    return output


def refine(baseline: list[dict], matches: list[dict | None], min_iou: float, alpha: float) -> list[dict]:
    output = []
    for row, match in zip(baseline, matches, strict=True):
        if match is None or match["iou"] < min_iou:
            output.append(row)
            continue
        item = dict(row)
        box = (1.0 - alpha) * np.asarray(row["bbox"], dtype=np.float64) + alpha * np.asarray(match["box"], dtype=np.float64)
        item["bbox"] = [int(round(float(value))) for value in box]
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--v13-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    for view in ("original", "grid_crops"):
        a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(a, b)
        folder = next(args.v13_cells.glob(f"v13_hires_localize_vfl_*_{view}"))
        expert = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
        matches = match_expert(baseline, expert)
        del expert
        data[view] = {"cell": a, "baseline": baseline, "matches": matches}
        print(f"LOADED {view} baseline={len(baseline)} matched={sum(item is not None for item in matches)}", flush=True)
    report = {"experiments": {}}
    for name, iou, alpha in [("baseline", 1.1, 0.0), ("vote_030_a025", 0.30, 0.25), ("vote_030_a050", 0.30, 0.50), ("vote_050_a025", 0.50, 0.25), ("vote_050_a050", 0.50, 0.50), ("vote_070_a025", 0.70, 0.25), ("vote_070_a050", 0.70, 0.50), ("replace_050", 0.50, 1.0)]:
        report["experiments"][name] = {}
        for view, cell in data.items():
            predictions = refine(cell["baseline"], cell["matches"], iou, alpha)
            metric = metrics50(predictions, cell["cell"]["ground_truth"], cell["cell"]["sizes"])
            report["experiments"][name][view] = metric
            print(f"{name} {view}: score={metric['score']:.6f} map50={metric['map50']:.6f} tp={metric['tp']}", flush=True)
    baseline = report["experiments"]["baseline"]
    for views in report["experiments"].values():
        for view, metric in views.items():
            metric["delta_vs_v7"] = {key: metric[key] - baseline[view][key] for key in ("score", "map50", "tp", "fp")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
