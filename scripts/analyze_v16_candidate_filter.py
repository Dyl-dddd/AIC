"""Audit and filter V12 out-of-fold candidates on both frozen dev views.

This program reads development labels only. It never reads Test labels or
creates a submission. It reports the exact scorer proxy and miss taxonomy.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_v12_perclass_ranker_oof import FOLDS, FEATURES, PERMITTED, fit
from scripts.analyze_v9_nonlinear_oof import add_context, matrix
from scripts.analyze_v9_support_oof import VIEW_WEIGHTS, add_support, initial_records
from scripts.analyze_varifocal_oof_transfer import load_groups
from steel_defect.geometry import box_iou


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_records(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    groups = load_groups(args.splits)
    cells, baselines, records = {}, {}, {}
    for view in VIEW_WEIGHTS:
        print(f"LOAD {view}", flush=True)
        a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(a, b)
        rows = initial_records(baseline, a["ground_truth"], groups)
        add_support(rows, read_json(args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json"), "vf")
        for prefix, folder in (
            ("p1", f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}"),
            ("p2", f"v9_model_b_quality_p2_last_224153cc09_{view}"),
        ):
            add_support(rows, read_json(args.v9_cells / folder / "predictions.json"), prefix)
        add_context(rows, a["sizes"])
        cells[view], baselines[view], records[view] = a, baseline, rows
    return cells, baselines, records


def score_oof(records: dict, betas: dict[str, float]) -> dict[str, list[dict]]:
    combined = [*records["original"], *records["grid_crops"]]
    for fold in range(FOLDS):
        for class_index, class_name in enumerate(PERMITTED):
            training = [row for row in combined if row["fold"] != fold and row["class_name"] == class_name]
            heldout = [row for row in combined if row["fold"] == fold and row["class_name"] == class_name]
            if {row["group"] for row in training} & {row["group"] for row in heldout}:
                raise ValueError("group leakage")
            model = fit(training, 3101 + 101 * fold + class_index)
            probabilities = model.predict_proba(matrix(heldout, FEATURES))[:, 1]
            for row, probability in zip(heldout, probabilities, strict=True):
                row["oof_probability"] = float(probability)
            print(f"FIT fold={fold} class={class_name}", flush=True)
    output = {}
    for view, rows in records.items():
        scored = []
        for row in rows:
            item = dict(row["prediction"])
            beta = float(betas[row["class_name"]])
            base = max(float(item["score"]), 1e-12)
            probability = max(float(row["oof_probability"]), 1e-12)
            item["score"] = base ** (1.0 - beta) * probability**beta
            scored.append(item)
        output[view] = scored
    return output


def top_k_per_image_class(predictions: list[dict], k: int) -> list[dict]:
    groups = defaultdict(list)
    for index, row in enumerate(predictions):
        groups[(row["image_id"], row["category_name"])].append(index)
    keep = set()
    for indices in groups.values():
        indices.sort(key=lambda index: float(predictions[index]["score"]), reverse=True)
        keep.update(indices[:k])
    return [row for index, row in enumerate(predictions) if index in keep]


def top_k_per_image(predictions: list[dict], k: int) -> list[dict]:
    groups = defaultdict(list)
    for index, row in enumerate(predictions):
        groups[row["image_id"]].append(index)
    keep = set()
    for indices in groups.values():
        indices.sort(key=lambda index: float(predictions[index]["score"]), reverse=True)
        keep.update(indices[:k])
    return [row for index, row in enumerate(predictions) if index in keep]


def support_tail_filter(predictions: list[dict], records: list[dict], quantile: float) -> list[dict]:
    support = [sum(row["features"][f"{prefix}_present"] for prefix in ("vf", "p1", "p2")) for row in records]
    unsupported = [float(pred["score"]) for pred, n in zip(predictions, support, strict=True) if n == 0]
    if not unsupported:
        return predictions
    floor = float(np.quantile(np.asarray(unsupported, dtype=np.float64), quantile))
    return [pred for pred, n in zip(predictions, support, strict=True) if n > 0 or float(pred["score"]) >= floor]


def miss_taxonomy(predictions: list[dict], ground_truth: dict) -> dict:
    by_image = defaultdict(list)
    for row in predictions:
        by_image[row["image_id"]].append(row)
    result = Counter()
    examples = []
    for image_id, classes in ground_truth.items():
        image_predictions = by_image[image_id]
        for class_name, gt_boxes in classes.items():
            same = [row for row in image_predictions if row["category_name"] == class_name]
            other = [row for row in image_predictions if row["category_name"] != class_name]
            for box in gt_boxes:
                target = np.asarray(box, dtype=np.float32)
                same_iou = float(np.max(box_iou(target, np.asarray([row["bbox"] for row in same], dtype=np.float32)))) if same else 0.0
                if same_iou >= 0.50:
                    result["covered"] += 1
                    continue
                other_iou = float(np.max(box_iou(target, np.asarray([row["bbox"] for row in other], dtype=np.float32)))) if other else 0.0
                if same_iou >= 0.30:
                    reason = "poor_localization"
                elif other_iou >= 0.50:
                    reason = "wrong_class"
                else:
                    reason = "missing_candidate"
                result[reason] += 1
                examples.append({"image_id": image_id, "category_name": class_name, "gt_box": box, "best_same_iou": same_iou, "best_other_iou": other_iou, "reason": reason})
    return {"counts": dict(result), "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--v9-cells", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--v12-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, baselines, records = make_records(args)
    spec = read_json(args.v12_results)
    v12 = score_oof(records, spec["decision"]["class_betas"])
    experiments = {"v7": baselines, "v12": v12}
    for k in (40, 80, 160, 320):
        experiments[f"top_class_{k}"] = {view: top_k_per_image_class(rows, k) for view, rows in v12.items()}
    for k in (80, 160, 320, 500):
        experiments[f"top_image_{k}"] = {view: top_k_per_image(rows, k) for view, rows in v12.items()}
    for q in (0.25, 0.50, 0.75, 0.90):
        experiments[f"unsupported_tail_{q:.2f}"] = {
            view: support_tail_filter(v12[view], records[view], q) for view in VIEW_WEIGHTS
        }
    report = {"experiments": {}, "miss_taxonomy": {}}
    for name, views in experiments.items():
        report["experiments"][name] = {
            view: metrics50(views[view], cells[view]["ground_truth"], cells[view]["sizes"])
            for view in VIEW_WEIGHTS
        }
        print(name, {view: {key: report["experiments"][name][view][key] for key in ("score", "map50", "tp", "fp")} for view in VIEW_WEIGHTS}, flush=True)
    for view in VIEW_WEIGHTS:
        report["miss_taxonomy"][view] = miss_taxonomy(v12[view], cells[view]["ground_truth"])
    for name in report["experiments"]:
        for view in VIEW_WEIGHTS:
            value = report["experiments"][name][view]
            base = report["experiments"]["v12"][view]
            value["delta_vs_v12"] = {key: value[key] - base[key] for key in ("score", "map50", "tp", "fp")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
