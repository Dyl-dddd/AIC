"""Test V13 tail rescue only on images with no V7/V12 predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--v13-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"views": {}}
    for view in ("original", "grid_crops"):
        a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(a, b)
        folder = next(args.v13_cells.glob(f"v13_hires_localize_vfl_*_{view}"))
        expert = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
        grouped = defaultdict(list)
        for row in expert:
            grouped[row["image_id"]].append(row)
        active = {row["image_id"] for row in baseline}
        missing = sorted(set(a["ground_truth"]) - active)
        truth_nonempty = sum(any(bool(boxes) for boxes in a["ground_truth"][image_id].values()) for image_id in missing)
        candidates = {image_id: sorted(grouped.get(image_id, []), key=lambda row: float(row["score"]), reverse=True) for image_id in missing}
        base = metrics50(baseline, a["ground_truth"], a["sizes"])
        trials = {}
        for k in (1, 3, 10):
            tail = min(float(row["score"]) for row in baseline) * 0.5
            rescue = []
            for image_id in missing:
                for index, row in enumerate(candidates[image_id][:k]):
                    item = dict(row)
                    item["score"] = tail * (1.0 - 0.01 * index)
                    rescue.append(item)
            metric = metrics50([*baseline, *rescue], a["ground_truth"], a["sizes"])
            trials[str(k)] = {"added": len(rescue), "score": metric["score"], "tp": metric["tp"], "fp": metric["fp"], "map50": metric["map50"], "delta_score": metric["score"] - base["score"]}
        result["views"][view] = {"empty_images": missing, "empty_images_with_gt": truth_nonempty, "baseline": {key: base[key] for key in ("score", "tp", "fp", "map50")}, "trials": trials}
        print(json.dumps({view: result["views"][view]}, ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
