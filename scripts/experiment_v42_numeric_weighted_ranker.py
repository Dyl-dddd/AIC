"""Predeclared V12 per-class numeric-domain weighting ablation.

The detector boxes are unchanged. A group-heldout ranker is fitted on the
frozen development candidates; no Test labels or pixels enter training.
This is an exploratory transfer audit, not an official-score estimate.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_v9_nonlinear_oof import GEOMETRY, add_context, matrix
from scripts.analyze_v9_support_oof import P1, P2, VF, add_support, initial_records
from scripts.analyze_varifocal_oof_transfer import balanced_weights, load_groups
from steel_defect.metrics import _match_class, average_precision


CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
QUALITY = ROOT / "runs/quality_aware_v1_fix3_results_received_20260928/quality_aware_v1_fix3_20260928_pipeline/audit/evaluation"
V9 = ROOT / "runs/v9_quality_localizer_results_received_20260930/runs/semifinal/v9_quality_localizer_pipeline/selection/cells"
SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
V12 = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
POLICY = ROOT / "runs/semifinal/v12_perclass_ranker_20261001/oof_results.json"
OUT = ROOT / "runs/semifinal/v42_numeric_weighted_ranker_20261004/results.json"
FEATURES = VF + P1 + P2 + GEOMETRY
CLASSES = ("zonglie", "mamianmakeng", "yanghuatiepi")
FACTORS = (1.0, 2.0)


def is_numeric(image_id: str) -> bool:
    return not image_id.split("::", 1)[0].startswith("C")


def fit(rows: list[dict], class_index: int, fold: int, factor: float) -> XGBClassifier:
    features = matrix(rows, FEATURES)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
    weights = balanced_weights(np.zeros(len(rows), dtype=np.int32), labels).astype(np.float32)
    weights *= np.asarray([factor if is_numeric(row["prediction"]["image_id"]) else 1.0
                           for row in rows], dtype=np.float32)
    model = XGBClassifier(
        n_estimators=180, max_depth=3, learning_rate=0.035,
        min_child_weight=12.0, subsample=0.82, colsample_bytree=0.88,
        reg_alpha=0.30, reg_lambda=10.0, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", max_bin=128,
        n_jobs=4, random_state=3101 + 101 * fold + class_index,
    )
    model.fit(features, labels, sample_weight=weights, verbose=False)
    return model


def score_rows(rows: list[dict], beta: float) -> list[dict]:
    result = []
    for row in rows:
        candidate = dict(row["prediction"])
        base = max(float(candidate["score"]), 1e-12)
        prob = max(float(row["probability"]), 1e-12)
        candidate["score"] = base ** (1 - beta) * prob ** beta
        result.append(candidate)
    return result


def ap(predictions: list[dict], gt: dict, class_name: str) -> tuple[float, int, int]:
    tp, fp, count = _match_class(predictions, gt, class_name, 0.5)
    return average_precision(tp, fp, count), int(tp.sum()), count


def main() -> None:
    groups = load_groups(SPLIT)
    records = {}
    ground_truth = {}
    baselines = {}
    for view in ("original", "grid_crops"):
        print("LOAD", view, flush=True)
        a = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        b = load_cell(CELLS / f"model_b_tta_s1280_{view}")
        base = fuse_v7(a, b)
        rows = initial_records(base, a["ground_truth"], groups)
        vf = json.loads((QUALITY / "varifocal" / f"varifocal_{view}_predictions.json").read_text(encoding="utf-8"))
        add_support(rows, vf, "vf")
        for prefix, folder in (
            ("p1", f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}"),
            ("p2", f"v9_model_b_quality_p2_last_224153cc09_{view}"),
        ):
            support = json.loads((V9 / folder / "predictions.json").read_text(encoding="utf-8"))
            add_support(rows, support, prefix)
        add_context(rows, a["sizes"])
        previous = json.loads((V12 / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        if len(rows) != len(previous):
            raise RuntimeError(f"V12 candidate count mismatch: {view}")
        for row, old in zip(rows, previous, strict=True):
            candidate = row["prediction"]
            if (candidate["image_id"], candidate["category_name"], candidate["bbox"]) != (
                old["image_id"], old["category_name"], old["bbox"]
            ):
                raise RuntimeError(f"V12 candidate alignment mismatch: {view}")
        records[view], ground_truth[view], baselines[view] = rows, a["ground_truth"], previous
        print("LOADED", view, len(rows), flush=True)
    policy = json.loads(POLICY.read_text(encoding="utf-8"))["decision"]["class_betas"]
    combined = records["original"] + records["grid_crops"]
    report = {"protocol": "5-fold source-group heldout; same V12 boxes/features/beta; numeric train weight 1 or 2",
              "warning": "Frozen dev is used repeatedly; official Test score unverified",
              "classes": {}}
    for class_name in CLASSES:
        class_index = ("jieba", "zonglie", "jiaza", "yiwuyaru", "huashang",
                       "mamianmakeng", "yanghuatiepi", "gunyin").index(class_name)
        target_rows = [row for row in combined if row["class_name"] == class_name]
        baseline = {}
        for view in records:
            base_rows = [row for row in baselines[view] if row["category_name"] == class_name]
            for source in ("numeric", "all"):
                chosen = [row for row in base_rows if source == "all" or is_numeric(row["image_id"])]
                gt = {k: v for k, v in ground_truth[view].items() if source == "all" or is_numeric(k)}
                baseline[f"{view}/{source}"] = ap(chosen, gt, class_name)[0]
        trials = {}
        for factor in FACTORS:
            for fold in range(5):
                train = [row for row in target_rows if row["fold"] != fold]
                heldout = [row for row in target_rows if row["fold"] == fold]
                if {row["group"] for row in train} & {row["group"] for row in heldout}:
                    raise RuntimeError("Group leakage")
                model = fit(train, class_index, fold, factor)
                probability = model.predict_proba(matrix(heldout, FEATURES))[:, 1]
                for row, value in zip(heldout, probability, strict=True):
                    row["probability"] = float(value)
                print("FIT", class_name, factor, fold, len(train), len(heldout), flush=True)
            trial = {}
            for view in records:
                view_rows = [row for row in records[view] if row["class_name"] == class_name]
                adjusted = score_rows(view_rows, float(policy[class_name]))
                for source in ("numeric", "all"):
                    chosen = [row for row in adjusted if source == "all" or is_numeric(row["image_id"])]
                    gt = {k: v for k, v in ground_truth[view].items() if source == "all" or is_numeric(k)}
                    value, tp, total = ap(chosen, gt, class_name)
                    key = f"{view}/{source}"
                    trial[key] = {"ap50": value, "delta_vs_v12": value - baseline[key],
                                  "tp": tp, "gt": total}
            trials[str(factor)] = trial
            print("TRIAL", class_name, factor, {k: round(v["delta_vs_v12"], 5)
                                                for k, v in trial.items()}, flush=True)
        report["classes"][class_name] = {"baseline_ap50": baseline, "trials": trials}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
