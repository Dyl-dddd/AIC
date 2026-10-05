"""Predeclared V12 zonglie geometry checks on frozen original and grid dev views."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell
from steel_defect.metrics import _match_class, average_precision


PRED_ROOT = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
CELL_ROOT = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OUT = ROOT / "runs/semifinal/v31_zonglie_extent_20261003"


def ap(rows: list[dict], gt: dict, threshold: float) -> dict:
    tp, fp, count = _match_class(rows, gt, "zonglie", threshold)
    return {"ap": average_precision(tp, fp, count), "gt": count,
            "tp": int(tp.sum()), "fp": int(fp.sum())}


def by_family(rows: list[dict], gt: dict, family: str, threshold: float) -> dict:
    if family == "all":
        return ap(rows, gt, threshold)
    wanted_c = family == "C"
    selected = [row for row in rows if row["image_id"].startswith("C") == wanted_c]
    selected_gt = {image_id: labels for image_id, labels in gt.items()
                   if image_id.startswith("C") == wanted_c}
    return ap(selected, selected_gt, threshold)


def change_box(row: dict, sizes: dict, height_scale: float, width_scale: float) -> dict:
    x1, y1, x2, y2 = row["bbox"]
    width, height = sizes[row["image_id"]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(1.0, (x2 - x1) * width_scale / 2)
    half_h = max(1.0, (y2 - y1) * height_scale / 2)
    box = [max(0, round(cx - half_w)), max(0, round(cy - half_h)),
           min(width, round(cx + half_w)), min(height, round(cy + half_h))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise RuntimeError(f"Invalid scaled box: {row['image_id']}, {box}")
    return {**row, "bbox": box}


def main() -> None:
    report = {}
    variants = [(1.25, 1.0), (1.5, 1.0), (2.0, 1.0), (2.5, 1.0),
                (1.25, 0.85), (1.5, 0.85), (2.0, 0.85), (2.5, 0.85)]
    for view in ("original", "grid_crops"):
        rows = json.loads((PRED_ROOT / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        rows = [row for row in rows if row["category_name"] == "zonglie"]
        cell = load_cell(CELL_ROOT / f"model_a_tta_s1024_{view}")
        gt, sizes = cell["ground_truth"], cell["sizes"]
        if any(row["image_id"] not in gt for row in rows):
            raise RuntimeError(f"Mismatched image ids: {view}")
        baseline = {family: {"0.5": by_family(rows, gt, family, 0.5),
                             "0.75": by_family(rows, gt, family, 0.75)}
                    for family in ("all", "numeric", "C")}
        result = {"baseline": baseline, "variants": []}
        for hscale, wscale in variants:
            changed = [change_box(row, sizes, hscale, wscale) for row in rows]
            scores = {family: {"0.5": by_family(changed, gt, family, 0.5),
                               "0.75": by_family(changed, gt, family, 0.75)}
                      for family in ("all", "numeric", "C")}
            result["variants"].append({"height_scale": hscale, "width_scale": wscale,
                                       "scores": scores,
                                       "delta_ap50": {family: scores[family]["0.5"]["ap"] - baseline[family]["0.5"]["ap"]
                                                      for family in ("all", "numeric", "C")}})
        report[view] = result
        print(view, "baseline", baseline, flush=True)
        for item in result["variants"]:
            print("scale", item["height_scale"], item["width_scale"],
                  "AP50 all", round(item["scores"]["all"]["0.5"]["ap"], 5),
                  "delta all/numeric/C", *(round(item["delta_ap50"][family], 5)
                                            for family in ("all", "numeric", "C")), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "results.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {target}", flush=True)


if __name__ == "__main__":
    main()
