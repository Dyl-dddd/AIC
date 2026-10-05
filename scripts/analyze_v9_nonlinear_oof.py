"""Strict group-OOF nonlinear ranking using VF and V9 support features."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_v9_support_oof import (
    P1,
    P2,
    VF,
    VIEW_WEIGHTS,
    add_support,
    initial_records,
)
from scripts.analyze_varifocal_oof_transfer import balanced_weights, load_groups
from steel_defect.classes import CLASS_NAMES


FOLDS = 5
PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
GEOMETRY = (
    "log_width_ratio", "log_height_ratio", "log_area_ratio", "log_aspect",
    "center_x", "center_y", "edge_distance", "image_rank", "image_class_rank",
    "log_image_count", "log_image_class_count",
)
CLASS_FEATURES = tuple(f"class_{name}" for name in PERMITTED)
FEATURE_SETS = {
    "nonlinear_vf": VF + GEOMETRY + CLASS_FEATURES,
    "nonlinear_vf_v9": VF + P1 + P2 + GEOMETRY + CLASS_FEATURES,
}
BLENDS = (0.10, 0.25, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00)


def add_context(records: list[dict], sizes: dict[str, tuple[int, int]]) -> None:
    by_image: dict[str, list[dict]] = defaultdict(list)
    by_image_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        prediction = row["prediction"]
        by_image[prediction["image_id"]].append(row)
        by_image_class[(prediction["image_id"], prediction["category_name"])].append(row)
    image_rank = {}
    class_rank = {}
    for rows in by_image.values():
        ordered = sorted(rows, key=lambda row: float(row["prediction"]["score"]), reverse=True)
        for rank, row in enumerate(ordered):
            image_rank[id(row)] = rank / max(len(ordered) - 1, 1)
    for rows in by_image_class.values():
        ordered = sorted(rows, key=lambda row: float(row["prediction"]["score"]), reverse=True)
        for rank, row in enumerate(ordered):
            class_rank[id(row)] = rank / max(len(ordered) - 1, 1)
    for row in records:
        prediction = row["prediction"]
        image_id = prediction["image_id"]
        class_name = prediction["category_name"]
        width, height = sizes[image_id]
        x1, y1, x2, y2 = map(float, prediction["bbox"])
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        width_ratio = box_width / max(width, 1)
        height_ratio = box_height / max(height, 1)
        center_x = (x1 + x2) / (2.0 * max(width, 1))
        center_y = (y1 + y2) / (2.0 * max(height, 1))
        edge_distance = min(center_x, 1.0 - center_x, center_y, 1.0 - center_y)
        features = row["features"]
        features.update({
            "log_width_ratio": math.log(max(width_ratio, 1e-8)),
            "log_height_ratio": math.log(max(height_ratio, 1e-8)),
            "log_area_ratio": math.log(max(width_ratio * height_ratio, 1e-12)),
            "log_aspect": math.log(max(box_width / box_height, 1e-8)),
            "center_x": center_x,
            "center_y": center_y,
            "edge_distance": edge_distance,
            "image_rank": image_rank[id(row)],
            "image_class_rank": class_rank[id(row)],
            "log_image_count": math.log(len(by_image[image_id])),
            "log_image_class_count": math.log(len(by_image_class[(image_id, class_name)])),
        })
        for name in PERMITTED:
            features[f"class_{name}"] = float(class_name == name)


def matrix(records: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[row["features"][name] for name in names] for row in records], dtype=np.float32)


def fit(records: list[dict], names: tuple[str, ...], seed: int) -> XGBClassifier:
    x = matrix(records, names)
    labels = np.asarray([row["label"] for row in records], dtype=np.int32)
    _, class_ids = np.unique([row["class_name"] for row in records], return_inverse=True)
    weights = balanced_weights(class_ids, labels).astype(np.float32)
    model = XGBClassifier(
        n_estimators=140,
        max_depth=3,
        learning_rate=0.04,
        min_child_weight=20.0,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=8.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=128,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x, labels, sample_weight=weights, verbose=False)
    return model


def apply(
    records: list[dict], names: tuple[str, ...], models: dict[int, XGBClassifier], blend: float,
) -> list[dict]:
    x = matrix(records, names)
    probabilities = np.zeros(len(records), dtype=np.float64)
    folds = np.asarray([row["fold"] for row in records])
    for fold, model in models.items():
        indices = np.flatnonzero(folds == fold)
        probabilities[indices] = model.predict_proba(x[indices])[:, 1]
    output = []
    for row, probability in zip(records, probabilities, strict=True):
        prediction = dict(row["prediction"])
        base = max(float(prediction["score"]), 1e-12)
        probability = max(float(probability), 1e-12)
        prediction["score"] = base ** (1.0 - blend) * probability ** blend
        output.append(prediction)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--v9-cells", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = load_groups(args.splits)
    cells, baselines, records = {}, {}, {}
    for view in VIEW_WEIGHTS:
        print(f"LOAD {view}", flush=True)
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        view_records = initial_records(baseline, cell_a["ground_truth"], groups)
        vf = json.loads((args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json").read_text())
        add_support(view_records, vf, "vf")
        for prefix, folder in (
            ("p1", f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}"),
            ("p2", f"v9_model_b_quality_p2_last_224153cc09_{view}"),
        ):
            predictions = json.loads((args.v9_cells / folder / "predictions.json").read_text())
            add_support(view_records, predictions, prefix)
            print(f"MATCH {view} {prefix}", flush=True)
        add_context(view_records, cell_a["sizes"])
        cells[view], baselines[view], records[view] = cell_a, baseline, view_records

    combined = [*records["original"], *records["grid_crops"]]
    models: dict[str, dict[int, XGBClassifier]] = {name: {} for name in FEATURE_SETS}
    audit = {}
    for fold in range(FOLDS):
        train = [row for row in combined if row["fold"] != fold]
        heldout = [row for row in combined if row["fold"] == fold]
        if {row["group"] for row in train} & {row["group"] for row in heldout}:
            raise RuntimeError("group leakage")
        audit[str(fold)] = {"train": len(train), "heldout": len(heldout)}
        for name, names in FEATURE_SETS.items():
            models[name][fold] = fit(train, names, 1701 + fold)
            print(f"FIT fold={fold} set={name}", flush=True)

    baseline_metrics = {
        view: metrics50(baselines[view], cells[view]["ground_truth"], cells[view]["sizes"])
        for view in VIEW_WEIGHTS
    }
    baseline_weighted = sum(VIEW_WEIGHTS[v] * baseline_metrics[v]["score"] for v in VIEW_WEIGHTS)
    results = {}
    for feature_set, names in FEATURE_SETS.items():
        for blend in BLENDS:
            tag = f"{feature_set}_b{int(blend * 100):03d}"
            views = {}
            for view in VIEW_WEIGHTS:
                predictions = apply(records[view], names, models[feature_set], blend)
                metric = metrics50(predictions, cells[view]["ground_truth"], cells[view]["sizes"])
                metric["score_delta"] = metric["score"] - baseline_metrics[view]["score"]
                metric["map50_delta"] = metric["map50"] - baseline_metrics[view]["map50"]
                views[view] = metric
            weighted = sum(VIEW_WEIGHTS[v] * views[v]["score"] for v in VIEW_WEIGHTS)
            results[tag] = {
                "feature_set": feature_set,
                "blend": blend,
                "weighted_delta": weighted - baseline_weighted,
                "robust_delta": min(views[v]["score_delta"] for v in VIEW_WEIGHTS),
                "views": views,
            }
            print(
                f"{tag}: weighted={weighted-baseline_weighted:+.6f} "
                f"original={views['original']['score_delta']:+.6f} "
                f"grid={views['grid_crops']['score_delta']:+.6f}",
                flush=True,
            )
    selected = max(results, key=lambda name: (results[name]["robust_delta"], results[name]["weighted_delta"]))
    selected_row = results[selected]
    deployment_names = FEATURE_SETS[selected_row["feature_set"]]
    deployment_model = fit(combined, deployment_names, 2701)
    deployment_path = args.output.with_name("deployment_model.ubj")
    deployment_path.parent.mkdir(parents=True, exist_ok=True)
    deployment_model.save_model(deployment_path)
    payload = {
        "protocol": {"folds": FOLDS, "group_oof": True, "model": "XGB depth3"},
        "fold_audit": audit,
        "baseline": baseline_metrics,
        "results": results,
        "decision": {
            "selected": selected,
            "release_allowed": results[selected]["weighted_delta"] >= 0.20
            and results[selected]["robust_delta"] >= 0.10
            and all(results[selected]["views"][v]["map50_delta"] >= 0.005 for v in VIEW_WEIGHTS),
        },
        "deployment": {
            "model": str(deployment_path),
            "feature_set": selected_row["feature_set"],
            "feature_names": list(deployment_names),
            "blend": selected_row["blend"],
            "fit_scope": "all frozen development rows after OOF protocol selection",
            "not_used_for_reported_oof_metrics": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
