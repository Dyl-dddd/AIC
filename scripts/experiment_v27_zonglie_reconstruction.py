"""Test collinear crack-fragment reconstruction on frozen V12 OOF predictions.

Adds a small number of longitudinal union boxes while preserving every original
prediction. This is a dev-only diagnostic; it never reads Test labels.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell
from steel_defect.metrics import _match_class, average_precision


PRED_ROOT = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
CELL_ROOT = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OUT = ROOT / "runs/semifinal/v27_zonglie_reconstruction_20261003"


def add_unions(rows: list[dict], sizes: dict, gap: int, factor: float, max_per_image: int = 15) -> list[dict]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["category_name"] == "zonglie" and row["score"] >= 0.005:
            by_image[row["image_id"]].append(row)
    additions = []
    for image_id, candidates in by_image.items():
        ranked = sorted(candidates, key=lambda r: r["score"], reverse=True)[:40]
        proposals = []
        width, height = sizes[image_id]
        for i, a in enumerate(ranked):
            ax1, ay1, ax2, ay2 = a["bbox"]
            aw, ah = ax2 - ax1, ay2 - ay1
            if not 6 <= aw <= 180 or ah < 50:
                continue
            for b in ranked[i + 1:]:
                bx1, by1, bx2, by2 = b["bbox"]
                bw, bh = bx2 - bx1, by2 - by1
                if not 6 <= bw <= 180 or bh < 50:
                    continue
                overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
                if overlap_x / min(aw, bw) < 0.45:
                    continue
                # One segment ends near the start of the next. Strong overlap
                # is already handled by original predictions and their NMS.
                first, second = ((ay1, ay2), (by1, by2)) if ay1 <= by1 else ((by1, by2), (ay1, ay2))
                vertical_gap = second[0] - first[1]
                if not -min(ah, bh) * 0.15 <= vertical_gap <= gap:
                    continue
                box = [max(0, min(ax1, bx1)), max(0, min(ay1, by1)),
                       min(width, max(ax2, bx2)), min(height, max(ay2, by2))]
                if box[3] - box[1] < 1.35 * max(ah, bh):
                    continue
                if box[3] - box[1] > 2400 or box[2] - box[0] > 250:
                    continue
                score = min(1.0, factor * min(float(a["score"]), float(b["score"])))
                proposals.append((score, tuple(box)))
        seen = set()
        for score, box in sorted(proposals, reverse=True):
            if box in seen:
                continue
            seen.add(box)
            additions.append({"image_id": image_id, "category_name": "zonglie", "bbox": list(box), "score": score})
            if len(seen) >= max_per_image:
                break
    return additions


def ap(rows: list[dict], gt: dict, family: str, threshold: float) -> dict:
    if family != "all":
        rows = [r for r in rows if r["image_id"].startswith("C") == (family == "C")]
        gt = {name: value for name, value in gt.items() if name.startswith("C") == (family == "C")}
    tp, fp, count = _match_class(rows, gt, "zonglie", threshold)
    return {"ap": average_precision(tp, fp, count), "gt": count,
            "tp": int(tp.sum()), "fp": int(fp.sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for view in ("original", "grid_crops"):
        print(f"Loading {view}", flush=True)
        predictions = json.loads((PRED_ROOT / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        cell = load_cell(CELL_ROOT / f"model_a_tta_s1024_{view}")
        gt, sizes = cell["ground_truth"], cell["sizes"]
        source = [r for r in predictions if r["category_name"] == "zonglie"]
        baseline = {family: {str(iou): ap(source, gt, family, iou) for iou in (0.5, 0.75)}
                    for family in ("all", "numeric", "C")}
        report[view] = {"baseline": baseline, "variants": {}}
        print(f"{view} baseline {baseline}", flush=True)
        for gap, factor in ((20, 0.75), (80, 0.75), (160, 0.75)):
            addition = [row for row in add_unions(source, sizes, gap, factor)
                        if not row["image_id"].startswith("C")]
            combined = source + addition
            result = {family: {str(iou): ap(combined, gt, family, iou) for iou in (0.5, 0.75)}
                      for family in ("all", "numeric", "C")}
            selected = sorted(combined, key=lambda r: r["score"], reverse=True)
            matched, _, _ = _match_class(selected, gt, "zonglie", 0.5)
            addition_ids = {id(row) for row in addition}
            new_tp = [{"prediction": row, "ground_truth": gt[row["image_id"]].get("zonglie", [])}
                      for row, is_tp in zip(selected, matched, strict=True)
                      if is_tp and id(row) in addition_ids]
            report[view]["variants"][f"gap{gap}_score{factor}"] = {
                "added": len(addition), "scores": result, "new_true_positive_examples": new_tp[:10]}
            print(f"{view} gap{gap} added={len(addition)} AP50 all={result['all']['0.5']['ap']:.5f} numeric={result['numeric']['0.5']['ap']:.5f}", flush=True)
    destination = OUT / "experiment.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
