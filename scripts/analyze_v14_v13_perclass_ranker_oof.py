"""Strict group-OOF evaluation of V12 rankers augmented with V13 localization evidence."""

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
from scripts.analyze_v9_nonlinear_oof import GEOMETRY, add_context, matrix
from scripts.analyze_v9_support_oof import P1, P2, VF, VIEW_WEIGHTS, add_support, initial_records
from scripts.analyze_varifocal_oof_transfer import balanced_weights, load_groups
from steel_defect.classes import CLASS_NAMES
from steel_defect.metrics import _match_class, average_precision


FOLDS = 5
PERMITTED = tuple(name for name in CLASS_NAMES if name != "qilie")
FEATURES = VF + P1 + P2 + GEOMETRY
LOC = ("loc_present", "loc_iou", "log_loc_score", "loc_log_advantage")
FEATURES = FEATURES + LOC
BETAS = (0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95)


def fit(rows: list[dict], seed: int) -> XGBClassifier:
    x = matrix(rows, FEATURES)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    classes = np.zeros(len(rows), dtype=np.int32)
    weights = balanced_weights(classes, labels).astype(np.float32)
    model = XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=12.0,
        subsample=0.82,
        colsample_bytree=0.88,
        reg_alpha=0.30,
        reg_lambda=10.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=128,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(x, labels, sample_weight=weights, verbose=False)
    return model


def predictions(rows: list[dict], betas: dict[str, float]) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row["prediction"])
        beta = float(betas.get(row["class_name"], 0.85))
        base = max(float(item["score"]), 1e-12)
        probability = max(float(row["perclass_probability"]), 1e-12)
        item["score"] = base ** (1.0 - beta) * probability ** beta
        output.append(item)
    return output


def class_ap(rows: list[dict], ground_truth: dict, class_name: str, beta: float) -> float:
    output = []
    for row in rows:
        if row["class_name"] != class_name:
            continue
        item = dict(row["prediction"])
        base = max(float(item["score"]), 1e-12)
        probability = max(float(row["perclass_probability"]), 1e-12)
        item["score"] = base ** (1.0 - beta) * probability ** beta
        output.append(item)
    tp, fp, count = _match_class(output, ground_truth, class_name, 0.5)
    return average_precision(tp, fp, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--v9-cells", type=Path, required=True)
    parser.add_argument("--v13-cells", type=Path, required=True)
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
        view_rows = initial_records(baseline, cell_a["ground_truth"], groups)
        vf = json.loads((args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json").read_text())
        add_support(view_rows, vf, "vf")
        for prefix, folder in (
            ("p1", f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}"),
            ("p2", f"v9_model_b_quality_p2_last_224153cc09_{view}"),
        ):
            rows = json.loads((args.v9_cells / folder / "predictions.json").read_text())
            add_support(view_rows, rows, prefix)
        loc_folder = next(args.v13_cells.glob(f"v13_hires_localize_vfl_*_{view}"))
        loc = json.loads((loc_folder / "predictions.json").read_text())
        add_support(view_rows, loc, "loc")
        del loc
        print(f"MATCH {view} loc", flush=True)
        add_context(view_rows, cell_a["sizes"])
        cells[view], baselines[view], records[view] = cell_a, baseline, view_rows

    combined = [*records["original"], *records["grid_crops"]]
    fold_audit: dict[str, dict] = defaultdict(dict)
    for fold in range(FOLDS):
        for class_index, class_name in enumerate(PERMITTED):
            train = [row for row in combined if row["fold"] != fold and row["class_name"] == class_name]
            heldout = [row for row in combined if row["fold"] == fold and row["class_name"] == class_name]
            if {row["group"] for row in train} & {row["group"] for row in heldout}:
                raise RuntimeError("group leakage")
            positives = sum(int(row["label"]) for row in train)
            if not positives or positives == len(train):
                raise ValueError(f"invalid label balance fold={fold} class={class_name}")
            model = fit(train, 3101 + 101 * fold + class_index)
            probability = model.predict_proba(matrix(heldout, FEATURES))[:, 1]
            for row, value in zip(heldout, probability, strict=True):
                row["perclass_probability"] = float(value)
            fold_audit[str(fold)][class_name] = {
                "train": len(train), "heldout": len(heldout), "positive_train": positives,
            }
            print(f"FIT fold={fold} class={class_name} train={len(train)} positive={positives}", flush=True)

    baseline_metrics = {
        view: metrics50(baselines[view], cells[view]["ground_truth"], cells[view]["sizes"])
        for view in VIEW_WEIGHTS
    }
    sweeps: dict[str, dict] = {}
    for class_name in PERMITTED:
        sweeps[class_name] = {}
        for beta in BETAS:
            views = {}
            for view in VIEW_WEIGHTS:
                ap = class_ap(records[view], cells[view]["ground_truth"], class_name, beta)
                base_ap = baseline_metrics[view]["per_class"][class_name]["ap50"]
                views[view] = {"ap50": ap, "ap50_delta": ap - base_ap}
            weighted_ap = sum(VIEW_WEIGHTS[v] * views[v]["ap50"] for v in VIEW_WEIGHTS)
            robust_delta = min(views[v]["ap50_delta"] for v in VIEW_WEIGHTS)
            sweeps[class_name][str(beta)] = {
                "beta": beta, "weighted_ap50": weighted_ap,
                "robust_ap50_delta": robust_delta, "views": views,
            }

    policy = {}
    for class_name in PERMITTED:
        candidates = list(sweeps[class_name].values())
        nonnegative = [row for row in candidates if row["robust_ap50_delta"] >= 0.0]
        eligible = nonnegative or candidates
        selected = max(eligible, key=lambda row: (row["weighted_ap50"], row["robust_ap50_delta"]))
        policy[class_name] = selected["beta"]

    final_views = {}
    for view in VIEW_WEIGHTS:
        metric = metrics50(predictions(records[view], policy), cells[view]["ground_truth"], cells[view]["sizes"])
        metric["score_delta"] = metric["score"] - baseline_metrics[view]["score"]
        metric["map50_delta"] = metric["map50"] - baseline_metrics[view]["map50"]
        final_views[view] = metric
    weighted_delta = sum(VIEW_WEIGHTS[v] * final_views[v]["score_delta"] for v in VIEW_WEIGHTS)
    robust_delta = min(final_views[v]["score_delta"] for v in VIEW_WEIGHTS)

    model_dir = args.output.with_suffix("").with_name(args.output.stem + "_models")
    model_dir.mkdir(parents=True, exist_ok=True)
    deployment = {}
    for class_index, class_name in enumerate(PERMITTED):
        rows = [row for row in combined if row["class_name"] == class_name]
        model = fit(rows, 5101 + class_index)
        path = model_dir / f"{class_name}.ubj"
        model.save_model(path)
        deployment[class_name] = str(path)
        print(f"DEPLOY class={class_name} rows={len(rows)}", flush=True)

    payload = {
        "protocol": {
            "folds": FOLDS,
            "group_oof": True,
            "model": "independent XGB depth3 per class",
            "delta_from_v12": "adds V13 localization support IoU/score features only",
        },
        "features": list(FEATURES),
        "fold_audit": fold_audit,
        "baseline": baseline_metrics,
        "sweeps": sweeps,
        "decision": {
            "class_betas": policy,
            "weighted_delta": weighted_delta,
            "robust_delta": robust_delta,
            "views": final_views,
            "beats_v12_oof": (
                weighted_delta > 0.6508259848003998
                and robust_delta > 0.6214837098429484
            ),
        },
        "deployment_models": deployment,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
