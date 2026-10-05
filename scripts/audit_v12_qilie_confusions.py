"""Check whether V12 non-q boxes already cover held-out q ground truth.

Read-only frozen-dev diagnosis. No relabeling, fitting, Test use, or submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
V12 = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_original_predictions.json"
OUT = ROOT / "runs/semifinal/v40_qilie_box_extent_20261004/v12_confusions.json"


def iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    lt = np.maximum(one[:2], many[:, :2])
    rb = np.minimum(one[2:], many[:, 2:])
    inter = np.prod(np.maximum(rb - lt, 0), axis=1)
    area_one = np.prod(np.maximum(one[2:] - one[:2], 0))
    area_many = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return inter / np.maximum(area_one + area_many - inter, 1e-9)


def main() -> None:
    frozen = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = json.loads(V12.read_text(encoding="utf-8"))
    q_names = [name for name in frozen["splits"]["dev"]
               if any(a[0] == "qilie" for a in frozen["records"][name]["annotations"])]
    report = {"protocol": "Frozen dev q GT vs unchanged V12 original-view OOF boxes",
              "q_images": {}, "summary": {"q_gt": 0, "gt_with_any_v12_iou50": 0}}
    for name in q_names:
        candidates = [r for r in rows if r["image_id"] == name]
        boxes = np.asarray([r["bbox"] for r in candidates], dtype=np.float64)
        target_rows = []
        for label in frozen["records"][name]["annotations"]:
            if label[0] != "qilie":
                continue
            gt = np.asarray(label[1:], dtype=np.float64)
            overlaps = iou(gt, boxes) if len(boxes) else np.asarray([])
            indices = np.argsort(overlaps)[-5:][::-1]
            nearest = [{"category_name": candidates[int(i)]["category_name"],
                        "bbox": candidates[int(i)]["bbox"],
                        "score": candidates[int(i)]["score"],
                        "iou": float(overlaps[int(i)])} for i in indices]
            target_rows.append({"gt": list(map(float, gt)), "nearest": nearest,
                                "v12_iou50": bool(len(overlaps) and overlaps.max() >= .5)})
            report["summary"]["q_gt"] += 1
            report["summary"]["gt_with_any_v12_iou50"] += target_rows[-1]["v12_iou50"]
        report["q_images"][name] = {"v12_candidate_count": len(candidates), "targets": target_rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    for name, data in report["q_images"].items():
        print(name, [(round(t["nearest"][0]["iou"], 3),
                      t["nearest"][0]["category_name"]) if t["nearest"] else None
                     for t in data["targets"]], flush=True)


if __name__ == "__main__":
    main()
