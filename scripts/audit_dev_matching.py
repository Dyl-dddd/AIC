"""Compare the repository matcher with conventional best-IoU VOC matching.

This is a diagnostic only; it never edits predictions or submission files.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import PERMITTED, load_cell
from steel_defect.geometry import box_iou
from steel_defect.metrics import _match_class, average_precision


def conventional(predictions: list[dict], gt: dict, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    selected = sorted((p for p in predictions if p["category_name"] == name),
                      key=lambda p: float(p["score"]), reverse=True)
    total_gt = sum(len(classes.get(name, [])) for classes in gt.values())
    matched = defaultdict(set)
    tp = np.zeros(len(selected), dtype=np.float64)
    fp = np.zeros(len(selected), dtype=np.float64)
    for i, pred in enumerate(selected):
        boxes = np.asarray(gt.get(pred["image_id"], {}).get(name, []), dtype=np.float32)
        if len(boxes):
            ious = box_iou(np.asarray(pred["bbox"], dtype=np.float32), boxes)
            best = int(np.argmax(ious))
            if float(ious[best]) >= 0.5 and best not in matched[pred["image_id"]]:
                tp[i] = 1
                matched[pred["image_id"]].add(best)
                continue
        fp[i] = 1
    return tp, fp, total_gt


def main() -> None:
    base = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
    cells = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
    for view, cell_name in (("original", "model_a_tta_s1024_original"),
                            ("grid_crops", "model_a_tta_s1024_grid_crops")):
        predictions = json.loads((base / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        gt = load_cell(cells / cell_name)["ground_truth"]
        reports = []
        for name in PERMITTED:
            old_tp, old_fp, old_n = _match_class(predictions, gt, name, 0.5)
            new_tp, new_fp, new_n = conventional(predictions, gt, name)
            assert old_n == new_n
            reports.append({"class": name, "gt": old_n,
                            "repo_ap50": average_precision(old_tp, old_fp, old_n),
                            "standard_ap50": average_precision(new_tp, new_fp, new_n),
                            "repo_tp": int(old_tp.sum()), "standard_tp": int(new_tp.sum())})
        print(json.dumps({"view": view, "repo_macro_ap50": np.mean([r["repo_ap50"] for r in reports]),
                          "standard_macro_ap50": np.mean([r["standard_ap50"] for r in reports]),
                          "classes": reports}, indent=2))


if __name__ == "__main__":
    main()
