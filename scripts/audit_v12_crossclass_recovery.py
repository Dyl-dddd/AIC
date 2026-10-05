"""Frozen-dev audit of V12 missed GT that overlap a wrong-class V12 box.

Read-only diagnostics; oracle overlap is not a deployable classifier and is not
an official score estimate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
V12 = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_original_predictions.json"
OUT = ROOT / "runs/semifinal/v40_qilie_box_extent_20261004/crossclass_audit.json"


def iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    if not len(many):
        return np.asarray([])
    lt = np.maximum(one[:2], many[:, :2])
    rb = np.minimum(one[2:], many[:, 2:])
    inter = np.prod(np.maximum(rb - lt, 0), axis=1)
    a = np.prod(np.maximum(one[2:] - one[:2], 0))
    b = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return inter / np.maximum(a + b - inter, 1e-9)


def main() -> None:
    frozen = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = json.loads(V12.read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    missed = []
    summary = Counter()
    for name in frozen["splits"]["dev"]:
        candidates = by_image[name]
        boxes = np.asarray([r["bbox"] for r in candidates], dtype=float)
        for class_name, *coords in frozen["records"][name]["annotations"]:
            overlaps = iou(np.asarray(coords, dtype=float), boxes)
            same = [(float(overlaps[i]), candidates[i]) for i in range(len(candidates))
                    if candidates[i]["category_name"] == class_name]
            best_same = max(same, key=lambda x: x[0], default=None)
            summary["gt_all"] += 1
            if best_same and best_same[0] >= .5:
                continue
            summary["gt_no_same_class_iou50"] += 1
            other = [(float(overlaps[i]), candidates[i]) for i in range(len(candidates))
                     if candidates[i]["category_name"] != class_name]
            best_other = max(other, key=lambda x: x[0], default=None)
            recoverable = bool(best_other and best_other[0] >= .5)
            summary["wrong_class_iou50"] += recoverable
            summary[f"miss_{class_name}"] += 1
            if recoverable:
                summary[f"recover_{class_name}_from_{best_other[1]['category_name']}"] += 1
            missed.append({"image_id": name, "gt_class": class_name, "gt": coords,
                           "best_same_iou": best_same[0] if best_same else 0.,
                           "best_other_iou": best_other[0] if best_other else 0.,
                           "best_other_class": best_other[1]["category_name"] if best_other else None,
                           "best_other_score": best_other[1]["score"] if best_other else None,
                           "numeric_source": name[0].isdigit(),
                           "recoverable_by_oracle_relabel": recoverable})
    report = {"protocol": "Frozen nine-class dev; V12 original-view OOF; per-GT any-box IoU oracle",
              "summary": dict(summary), "missed": missed,
              "warning": "Oracle coverage is not an implementable rule and may overstate attainable recall."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(summary), ensure_ascii=False), flush=True)
    print("NUMERIC_RECOVERABLE", sum(r["numeric_source"] and r["recoverable_by_oracle_relabel"] for r in missed), flush=True)


if __name__ == "__main__":
    main()
