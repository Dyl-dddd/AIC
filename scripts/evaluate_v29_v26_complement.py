"""Compare V29 V26 proposals to frozen V12 OOF on the same original dev images."""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell
from steel_defect.geometry import box_iou
from steel_defect.metrics import _match_class, average_precision, validate_predictions


V29 = ROOT / "runs/semifinal/v29_results_received_20261003"
V12 = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_original_predictions.json"
CELL = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells/model_a_tta_s1024_original"
OUT = ROOT / "runs/semifinal/v29_v26_complement_eval_20261003"
CLASSES = ["jieba", "zonglie", "jiaza", "yiwuyaru", "huashang", "mamianmakeng", "yanghuatiepi", "gunyin"]


def score(rows: list[dict], gt: dict, name: str, iou: float, floor: float = 0.0) -> dict:
    tp, fp, total = _match_class(rows, gt, name, iou, floor)
    return {"ap": average_precision(tp, fp, total), "gt": total,
            "tp": int(tp.sum()), "fp": int(fp.sum()), "predictions": len(tp)}


def matched_gt(rows: list[dict], gt: dict, name: str, floor: float) -> set[tuple[str, int]]:
    grouped = defaultdict(list)
    for row in rows:
        if row["category_name"] == name and row["score"] >= floor:
            grouped[row["image_id"]].append(row["bbox"])
    hits = set()
    for image_id, per_class in gt.items():
        candidates = grouped[image_id]
        if not candidates:
            continue
        boxes = np.asarray(candidates, dtype=np.float32)
        for index, truth in enumerate(per_class.get(name, [])):
            if np.max(box_iou(np.asarray(truth, dtype=np.float32), boxes)) >= 0.5:
                hits.add((image_id, index))
    return hits


def consensus_rows(v12_rows: list[dict], v26_rows: list[dict], class_name: str,
                   v26_floor: float = 0.05) -> tuple[list[dict], np.ndarray]:
    grouped = defaultdict(list)
    for row in v26_rows:
        if row["category_name"] == class_name and row["score"] >= v26_floor:
            grouped[row["image_id"]].append(row["bbox"])
    arrays = {image_id: np.asarray(boxes, dtype=np.float32) for image_id, boxes in grouped.items()}
    selected = [row for row in v12_rows if row["category_name"] == class_name]
    support = np.zeros(len(selected), dtype=np.float32)
    for index, row in enumerate(selected):
        boxes = arrays.get(row["image_id"])
        if boxes is not None:
            support[index] = float(np.max(box_iou(np.asarray(row["bbox"], dtype=np.float32), boxes)))
    return selected, support


def main() -> None:
    manifest = json.loads((V29 / "RESULT.json").read_text(encoding="utf-8"))
    files = sorted((V29 / "dev").glob("*.json.gz"))
    if len(files) != manifest["dev_images"] or len(files) != 459:
        raise RuntimeError("V29 result is incomplete")
    v26 = []
    image_ids = set()
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["signature"] != manifest["signature"]:
            raise RuntimeError(f"Cache signature mismatch: {path}")
        image_ids.add(payload["image_id"])
        v26.extend(payload["predictions"])
    if len(image_ids) != 459 or len(v26) != manifest["predictions"]:
        raise RuntimeError("Image identity or prediction count mismatch")
    cell = load_cell(CELL)
    gt, sizes = cell["ground_truth"], cell["sizes"]
    if image_ids != set(gt):
        raise RuntimeError("V12 and V26 dev image identities differ")
    v12 = json.loads(V12.read_text(encoding="utf-8"))
    validate_predictions(v26, gt, CLASSES, sizes)
    validate_predictions(v12, gt, CLASSES, sizes)
    report = {"manifest": manifest, "v12_predictions": len(v12), "v26_predictions": len(v26),
              "counts_v26": dict(Counter(row["category_name"] for row in v26)), "classes": {},
              "coverage": {}, "coverage_by_class_005": {},
              "zonglie_addition_trials": [], "yanghuatiepi_addition_trials": [],
              "zonglie_consensus_trials": []}
    for name in CLASSES:
        report["classes"][name] = {
            model: {str(iou): score(rows, gt, name, iou) for iou in (0.5, 0.75)}
            for model, rows in (("v12", v12), ("v26", v26))
        }
        print(name, "V12", round(report["classes"][name]["v12"]["0.5"]["ap"], 4),
              "V26", round(report["classes"][name]["v26"]["0.5"]["ap"], 4), flush=True)
        old = matched_gt(v12, gt, name, 0.05)
        new = matched_gt(v26, gt, name, 0.05)
        report["coverage_by_class_005"][name] = {
            "v12": len(old), "v26": len(new), "v26_only": len(new - old), "v12_only": len(old - new)
        }
    for floor in (0.0, 0.01, 0.05, 0.2):
        old = matched_gt(v12, gt, "zonglie", floor)
        new = matched_gt(v26, gt, "zonglie", floor)
        report["coverage"][str(floor)] = {"v12": len(old), "v26": len(new),
                                          "v26_only": len(new - old), "v12_only": len(old - new),
                                          "v26_only_images": sorted({item[0] for item in new - old})}
    v12_z = [row for row in v12 if row["category_name"] == "zonglie"]
    v26_z = [row for row in v26 if row["category_name"] == "zonglie"]
    baseline = report["classes"]["zonglie"]["v12"]["0.5"]["ap"]
    for floor in (0.05, 0.2):
        for factor in (0.1, 0.3, 0.6):
            additions = [{**row, "score": float(row["score"]) * factor}
                         for row in v26_z if row["score"] >= floor]
            metrics = score(v12_z + additions, gt, "zonglie", 0.5)
            report["zonglie_addition_trials"].append({"v26_floor": floor, "v26_score_factor": factor,
                                                      "added": len(additions), "ap50": metrics["ap"],
                                                      "delta_vs_v12": metrics["ap"] - baseline})
            print("union", floor, factor, "AP50", round(metrics["ap"], 5), flush=True)
    v12_y = [row for row in v12 if row["category_name"] == "yanghuatiepi"]
    v26_y = [row for row in v26 if row["category_name"] == "yanghuatiepi"]
    baseline_y = report["classes"]["yanghuatiepi"]["v12"]["0.5"]["ap"]
    for floor in (0.05, 0.2):
        for factor in (0.1, 0.3, 0.6):
            additions = [{**row, "score": float(row["score"]) * factor}
                         for row in v26_y if row["score"] >= floor]
            metrics = score(v12_y + additions, gt, "yanghuatiepi", 0.5)
            report["yanghuatiepi_addition_trials"].append({"v26_floor": floor, "v26_score_factor": factor,
                                                            "added": len(additions), "ap50": metrics["ap"],
                                                            "delta_vs_v12": metrics["ap"] - baseline_y})
            print("yanghuatiepi_union", floor, factor, "AP50", round(metrics["ap"], 5), flush=True)
    selected, support = consensus_rows(v12, v26, "zonglie")
    for overlap_floor in (0.3, 0.5):
        agreement = support >= overlap_floor
        for boost in (0.5, 1.0, 2.0):
            rescored = [{**row, "score": min(1.0, float(row["score"]) * (1.0 + boost * float(ok)))}
                        for row, ok in zip(selected, agreement, strict=True)]
            ap50 = score(rescored, gt, "zonglie", 0.5)["ap"]
            ap75 = score(rescored, gt, "zonglie", 0.75)["ap"]
            report["zonglie_consensus_trials"].append({"overlap_floor": overlap_floor, "boost": boost,
                                                       "supported_v12_boxes": int(agreement.sum()),
                                                       "ap50": ap50, "ap75": ap75,
                                                       "delta_ap50": ap50 - baseline})
            print("consensus", overlap_floor, boost, "AP50", round(ap50, 5), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {target}", flush=True)


if __name__ == "__main__":
    main()
