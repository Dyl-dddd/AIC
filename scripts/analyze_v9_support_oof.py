"""Five-fold group OOF ranking with Varifocal plus V9 support evidence.

The detector membership and coordinates stay identical to V7.  Only scores are
re-ranked.  Models are fitted on four official groups folds and evaluated on the
held-out fold, so the reported gains are out-of-fold rather than in-sample.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_varifocal_oof_transfer import (
    balanced_weights,
    fold_for_group,
    label_prediction,
    load_groups,
    quality_matches,
    source_key,
)


FOLDS = 5
L2 = 0.10
VIEW_WEIGHTS = {"original": 488 / 788, "grid_crops": 300 / 788}
EXPECTED_V7 = {"original": 68.70285282028146, "grid_crops": 69.11447644900377}
VF = ("log_base_score", "vf_present", "vf_iou", "log_vf_score", "vf_log_advantage")
P1 = ("p1_present", "p1_iou", "log_p1_score", "p1_log_advantage")
P2 = ("p2_present", "p2_iou", "log_p2_score", "p2_log_advantage")
FEATURE_SETS = {
    "vf": VF,
    "vf_p1": VF + P1,
    "vf_p2": VF + P2,
    "vf_p1p2": VF + P1 + P2,
}
BLENDS = (0.25, 0.50, 0.75, 1.00)


def initial_records(baseline: list[dict], ground_truth: dict, groups: dict[str, str]) -> list[dict]:
    output = []
    for prediction in baseline:
        source = source_key(prediction["image_id"])
        group = groups[source]
        base = max(float(prediction["score"]), 1e-12)
        output.append({
            "prediction": prediction,
            "features": {"log_base_score": math.log(base)},
            "label": label_prediction(prediction, ground_truth[prediction["image_id"]]),
            "class_name": prediction["category_name"],
            "group": group,
            "fold": fold_for_group(group),
        })
    return output


def add_support(records: list[dict], predictions: list[dict], prefix: str) -> None:
    baseline = [row["prediction"] for row in records]
    matches = quality_matches(baseline, predictions)
    for row, match in zip(records, matches, strict=True):
        base = max(float(row["prediction"]["score"]), 1e-12)
        score = max(float(match["vf_score"]), 1e-12)
        row["features"].update({
            f"{prefix}_present": float(match["vf_present"]),
            f"{prefix}_iou": float(match["vf_iou"]),
            f"log_{prefix}_score": math.log(score),
            f"{prefix}_log_advantage": math.log(score) - math.log(base),
        })


def fit(records: list[dict], names: tuple[str, ...]) -> dict:
    matrix = np.asarray([[row["features"][name] for name in names] for row in records])
    labels = np.asarray([row["label"] for row in records], dtype=np.float64)
    classes, class_ids = np.unique([row["class_name"] for row in records], return_inverse=True)
    del classes
    weights = balanced_weights(class_ids, labels)
    means = matrix.mean(axis=0)
    scales = np.maximum(matrix.std(axis=0), 1e-6)
    standardized = (matrix - means) / scales
    denominator = float(weights.sum())

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = parameters[0] + standardized @ parameters[1:]
        loss = float(np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits))) / denominator
        loss += 0.5 * L2 * float(np.dot(parameters[1:], parameters[1:]))
        probabilities = np.where(
            logits >= 0, 1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        )
        residual = weights * (probabilities - labels) / denominator
        gradient = np.r_[residual.sum(), standardized.T @ residual + L2 * parameters[1:]]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(len(names) + 1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None)] + [(0.0, None)] * len(names),
        options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return {
        "names": names,
        "mean": means,
        "scale": scales,
        "intercept": float(result.x[0]),
        "coefficients": result.x[1:],
        "samples": len(records),
        "positives": int(labels.sum()),
    }


def probability(row: dict, profile: dict) -> float:
    vector = np.asarray([row["features"][name] for name in profile["names"]])
    standardized = (vector - profile["mean"]) / profile["scale"]
    logit = profile["intercept"] + float(standardized @ profile["coefficients"])
    return 1.0 / (1.0 + math.exp(-max(min(logit, 40.0), -40.0)))


def apply(records: list[dict], profiles: dict[int, dict], blend: float) -> list[dict]:
    output = []
    for row in records:
        prediction = dict(row["prediction"])
        base = max(float(prediction["score"]), 1e-12)
        calibrated = max(probability(row, profiles[row["fold"]]), 1e-12)
        prediction["score"] = base ** (1.0 - blend) * calibrated ** blend
        output.append(prediction)
    return output


def apply_residual(
    records: list[dict], vf_profiles: dict[int, dict], joint_profiles: dict[int, dict],
    alpha: float, vf_blend: float = 0.75,
) -> list[dict]:
    """Add a small V9 residual without discarding the protected VF ranking."""
    output = []
    for row in records:
        prediction = dict(row["prediction"])
        base = max(float(prediction["score"]), 1e-12)
        vf_probability = max(probability(row, vf_profiles[row["fold"]]), 1e-12)
        joint_probability = max(probability(row, joint_profiles[row["fold"]]), 1e-12)
        prediction["score"] = (
            base ** (1.0 - vf_blend)
            * vf_probability ** (vf_blend - alpha)
            * joint_probability ** alpha
        )
        output.append(prediction)
    return output


def serialize_profile(profile: dict) -> dict:
    return {
        "names": list(profile["names"]),
        "mean": profile["mean"].tolist(),
        "scale": profile["scale"].tolist(),
        "intercept": profile["intercept"],
        "coefficients": profile["coefficients"].tolist(),
        "samples": profile["samples"],
        "positives": profile["positives"],
    }


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
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if not math.isclose(metric["score"], EXPECTED_V7[view], abs_tol=1e-10):
            raise ValueError(f"V7 baseline mismatch: {view} {metric['score']}")
        view_records = initial_records(baseline, cell_a["ground_truth"], groups)
        vf = json.loads((args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json").read_text())
        add_support(view_records, vf, "vf")
        del vf
        for prefix, folder in (
            ("p1", f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}"),
            ("p2", f"v9_model_b_quality_p2_last_224153cc09_{view}"),
        ):
            predictions = json.loads((args.v9_cells / folder / "predictions.json").read_text())
            add_support(view_records, predictions, prefix)
            del predictions
            print(f"MATCH {view} {prefix}", flush=True)
        cells[view], baselines[view], records[view] = cell_a, baseline, view_records

    combined = [*records["original"], *records["grid_crops"]]
    profiles: dict[str, dict[int, dict]] = {name: {} for name in FEATURE_SETS}
    audit = {}
    for fold in range(FOLDS):
        train = [row for row in combined if row["fold"] != fold]
        heldout = [row for row in combined if row["fold"] == fold]
        train_groups = {row["group"] for row in train}
        heldout_groups = {row["group"] for row in heldout}
        if train_groups & heldout_groups:
            raise RuntimeError("group leakage")
        audit[str(fold)] = {"train": len(train), "heldout": len(heldout), "heldout_groups": len(heldout_groups)}
        for feature_set, names in FEATURE_SETS.items():
            profiles[feature_set][fold] = fit(train, names)
            print(f"FIT fold={fold} set={feature_set}", flush=True)

    baseline_metrics = {
        view: metrics50(baselines[view], cells[view]["ground_truth"], cells[view]["sizes"])
        for view in VIEW_WEIGHTS
    }
    baseline_weighted = sum(VIEW_WEIGHTS[v] * baseline_metrics[v]["score"] for v in VIEW_WEIGHTS)
    results = {}
    for feature_set in FEATURE_SETS:
        for blend in BLENDS:
            tag = f"{feature_set}_b{int(blend * 100):03d}"
            views = {}
            for view in VIEW_WEIGHTS:
                predictions = apply(records[view], profiles[feature_set], blend)
                metric = metrics50(predictions, cells[view]["ground_truth"], cells[view]["sizes"])
                metric["score_delta"] = metric["score"] - baseline_metrics[view]["score"]
                metric["map50_delta"] = metric["map50"] - baseline_metrics[view]["map50"]
                views[view] = metric
            weighted = sum(VIEW_WEIGHTS[v] * views[v]["score"] for v in VIEW_WEIGHTS)
            results[tag] = {
                "feature_set": feature_set,
                "blend": blend,
                "weighted_score": weighted,
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
    for joint_set in ("vf_p1", "vf_p2", "vf_p1p2"):
        for alpha in (0.025, 0.05, 0.10, 0.20):
            tag = f"vf_b075_{joint_set}_residual_a{int(alpha * 1000):03d}"
            views = {}
            for view in VIEW_WEIGHTS:
                predictions = apply_residual(
                    records[view], profiles["vf"], profiles[joint_set], alpha
                )
                metric = metrics50(predictions, cells[view]["ground_truth"], cells[view]["sizes"])
                metric["score_delta"] = metric["score"] - baseline_metrics[view]["score"]
                metric["map50_delta"] = metric["map50"] - baseline_metrics[view]["map50"]
                views[view] = metric
            weighted = sum(VIEW_WEIGHTS[v] * views[v]["score"] for v in VIEW_WEIGHTS)
            results[tag] = {
                "feature_set": "vf",
                "residual_joint": joint_set,
                "blend": 0.75,
                "residual_alpha": alpha,
                "weighted_score": weighted,
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
    deployment_profile = fit(combined, FEATURE_SETS[results[selected]["feature_set"]])
    residual_profile = None
    if "residual_joint" in results[selected]:
        residual_profile = fit(combined, FEATURE_SETS[results[selected]["residual_joint"]])
    payload = {
        "protocol": {"folds": FOLDS, "l2": L2, "group_oof": True},
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
            "feature_set": results[selected]["feature_set"],
            "blend": results[selected]["blend"],
            "profile": serialize_profile(deployment_profile),
            "residual_joint": results[selected].get("residual_joint"),
            "residual_alpha": results[selected].get("residual_alpha"),
            "residual_profile": serialize_profile(residual_profile) if residual_profile else None,
        },
        "profiles": {
            name: {str(fold): serialize_profile(profile) for fold, profile in fold_profiles.items()}
            for name, fold_profiles in profiles.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
