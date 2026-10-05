"""Five-fold official-group OOF calibration of Varifocal evidence onto V7 anchors."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import enrich_view as enrich_safe_view, fuse_v7
from scripts.analyze_varifocal_score_transfer import iou_matrix
from steel_defect.classes import CLASS_NAMES, CLASS_TO_ID
from steel_defect.geometry import box_iou
from steel_defect.safe_r2 import FEATURE_NAMES as SAFE_FEATURE_NAMES


FOLDS = 5
L2 = 0.10
MATCH_FLOOR = 0.50
VIEW_WEIGHTS = {"original": 488 / 788, "grid_crops": 300 / 788}
EXPECTED_V7 = {"original": 68.70285282028146, "grid_crops": 69.11447644900377}
PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
BASE_FEATURES = ("log_base_score",)
VF_FEATURES = (
    "log_base_score", "vf_present", "vf_iou", "log_vf_score", "vf_log_advantage",
)
CLASS_FEATURES = VF_FEATURES + tuple(f"class_{name}" for name in PERMITTED[:-1])
COMBINED_FEATURES = VF_FEATURES + tuple(
    f"safe_{name}" for name in SAFE_FEATURE_NAMES if name != "log_base_score"
)
FEATURE_SETS = {
    "base_only": BASE_FEATURES,
    "base_vf": VF_FEATURES,
    "base_vf_class": CLASS_FEATURES,
    "base_vf_safe": COMBINED_FEATURES,
}
VARIANTS = {
    "A0_v7": (None, 0.0),
    "A1_base_only_b25": ("base_only", 0.25),
    "A2_vf_b25": ("base_vf", 0.25),
    "A3_vf_b50": ("base_vf", 0.50),
    "A3b_vf_b75": ("base_vf", 0.75),
    "A3c_vf_b100": ("base_vf", 1.00),
    "A4_vf_class_b25": ("base_vf_class", 0.25),
    "A5_vf_safe_b50": ("base_vf_safe", 0.50),
    "A6_vf_safe_b75": ("base_vf_safe", 0.75),
}


def source_key(image_id: str) -> str:
    return image_id.split("::", 1)[0]


def fold_for_group(group: str) -> int:
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def load_groups(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {name: str(row["group"]) for name, row in payload["records"].items()}


def label_prediction(prediction: dict, ground_truth: dict) -> int:
    boxes = ground_truth.get(prediction["category_name"], [])
    if not boxes:
        return 0
    overlaps = box_iou(
        np.asarray(prediction["bbox"], dtype=np.float32),
        np.asarray(boxes, dtype=np.float32),
    )
    return int(float(np.max(overlaps)) >= 0.50)


def quality_matches(baseline: list[dict], quality: list[dict]) -> list[dict]:
    base_groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    quality_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for index, item in enumerate(baseline):
        base_groups[(item["image_id"], item["category_name"])].append((index, item))
    for item in quality:
        quality_groups[(item["image_id"], item["category_name"])].append(item)
    matches = [{"vf_present": 0.0, "vf_iou": 0.0, "vf_score": 1e-12} for _ in baseline]
    for key, base_items in base_groups.items():
        quality_items = quality_groups.get(key, [])
        if not quality_items:
            continue
        left = np.asarray([item[1]["bbox"] for item in base_items], dtype=np.float32)
        right = np.asarray([item["bbox"] for item in quality_items], dtype=np.float32)
        # Limit the temporary IoU tensor for dense images without changing
        # the nearest-quality-box match for any baseline prediction.
        for start in range(0, len(left), 64):
            matrix = iou_matrix(left[start:start + 64], right)
            best = matrix.argmax(axis=1)
            overlaps = matrix[np.arange(len(best)), best]
            for offset, overlap in enumerate(overlaps):
                if float(overlap) < MATCH_FLOOR:
                    continue
                target = base_items[start + offset][0]
                matches[target] = {
                    "vf_present": 1.0,
                    "vf_iou": float(overlap),
                    "vf_score": float(quality_items[int(best[offset])]["score"]),
                }
    return matches


def enrich(
    baseline: list[dict], quality: list[dict], ground_truth: dict,
    groups: dict[str, str], view: str,
) -> list[dict]:
    matches = quality_matches(baseline, quality)
    records = []
    for prediction, match in zip(baseline, matches):
        source = source_key(prediction["image_id"])
        group = groups[source]
        base = max(float(prediction["score"]), 1e-12)
        vf = max(float(match["vf_score"]), 1e-12)
        features = {
            "log_base_score": math.log(base),
            "vf_present": match["vf_present"],
            "vf_iou": match["vf_iou"],
            "log_vf_score": math.log(vf),
            "vf_log_advantage": math.log(vf) - math.log(base),
        }
        for name in PERMITTED[:-1]:
            features[f"class_{name}"] = float(prediction["category_name"] == name)
        records.append({
            "prediction": prediction,
            "features": features,
            "label": label_prediction(prediction, ground_truth[prediction["image_id"]]),
            "class_id": CLASS_TO_ID[prediction["category_name"]],
            "group": group,
            "fold": fold_for_group(group),
            "view": view,
        })
    return records


def balanced_weights(classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    strata = []
    for class_id in sorted(set(classes.tolist())):
        for label in (0, 1):
            indices = np.flatnonzero((classes == class_id) & (labels == label))
            if len(indices):
                strata.append(indices)
    for indices in strata:
        weights[indices] = 1.0 / len(indices)
    if not strata or not np.all(weights > 0):
        raise ValueError("incomplete class/label strata")
    return weights * len(weights) / float(weights.sum())


def fit(records: list[dict], names: tuple[str, ...]) -> dict:
    matrix = np.asarray([[row["features"][name] for name in names] for row in records])
    labels = np.asarray([row["label"] for row in records], dtype=np.float64)
    classes = np.asarray([row["class_id"] for row in records], dtype=np.int64)
    weights = balanced_weights(classes, labels)
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

    bounds = [(None, None)] + [
        (0.0, None) if name in {
            "log_base_score", "vf_present", "vf_iou", "log_vf_score", "vf_log_advantage"
        } or name in {
            "safe_ab_presence", "safe_ab_iou", "safe_source_count",
            "safe_global_local_coherence",
        } else (None, 0.0) if name in {
            "safe_dispersion", "safe_edge_only",
        } else (None, None)
        for name in names
    ]
    result = minimize(
        objective, np.zeros(len(names) + 1), method="L-BFGS-B", jac=True,
        bounds=bounds, options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7},
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
    value = (vector - profile["mean"]) / profile["scale"]
    logit = profile["intercept"] + float(value @ profile["coefficients"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = load_groups(args.splits)
    cells, baselines, records = {}, {}, {}
    for view in VIEW_WEIGHTS:
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if not math.isclose(metric["score"], EXPECTED_V7[view], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"V7 baseline mismatch: {view} {metric['score']}")
        quality = json.loads((
            args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json"
        ).read_text())
        cells[view], baselines[view] = cell_a, baseline
        records[view] = enrich(baseline, quality, cell_a["ground_truth"], groups, view)
        safe_records = enrich_safe_view(cell_a, cell_b, baseline, groups, view)
        if len(safe_records) != len(records[view]):
            raise RuntimeError("SAFE-R2/VF feature rows differ in length")
        for vf_row, safe_row in zip(records[view], safe_records):
            if vf_row["prediction"] != safe_row["prediction"]:
                raise RuntimeError("SAFE-R2/VF feature rows are not aligned")
            for index, name in enumerate(SAFE_FEATURE_NAMES):
                if name != "log_base_score":
                    vf_row["features"][f"safe_{name}"] = float(safe_row["features"][index])
        print(f"ENRICH {view}: {len(records[view])}", flush=True)
    combined = [*records["original"], *records["grid_crops"]]
    profiles: dict[str, dict[int, dict]] = {name: {} for name in FEATURE_SETS}
    audit = {}
    for fold in range(FOLDS):
        train = [row for row in combined if row["fold"] != fold]
        heldout_groups = {row["group"] for row in combined if row["fold"] == fold}
        if heldout_groups & {row["group"] for row in train}:
            raise RuntimeError("group leakage")
        audit[str(fold)] = {
            "train": len(train),
            "heldout": sum(row["fold"] == fold for row in combined),
            "heldout_groups": len(heldout_groups),
        }
        for feature_set, names in FEATURE_SETS.items():
            profiles[feature_set][fold] = fit(train, names)
            print(f"FIT fold={fold} set={feature_set}", flush=True)
    results = {}
    baseline_metrics = {
        view: metrics50(baselines[view], cells[view]["ground_truth"], cells[view]["sizes"])
        for view in VIEW_WEIGHTS
    }
    baseline_weighted = sum(VIEW_WEIGHTS[v] * baseline_metrics[v]["score"] for v in VIEW_WEIGHTS)
    for variant, (feature_set, blend) in VARIANTS.items():
        views = {}
        for view in VIEW_WEIGHTS:
            predictions = baselines[view] if feature_set is None else apply(
                records[view], profiles[feature_set], blend
            )
            metric = metrics50(predictions, cells[view]["ground_truth"], cells[view]["sizes"])
            views[view] = {
                **metric,
                "score_delta": metric["score"] - baseline_metrics[view]["score"],
                "map50_delta": metric["map50"] - baseline_metrics[view]["map50"],
            }
        weighted = sum(VIEW_WEIGHTS[v] * views[v]["score"] for v in VIEW_WEIGHTS)
        results[variant] = {
            "weighted_score": weighted,
            "weighted_delta": weighted - baseline_weighted,
            "views": views,
        }
        print(
            f"{variant}: weighted_delta={weighted-baseline_weighted:+.9f} "
            f"original={views['original']['score_delta']:+.9f} "
            f"grid={views['grid_crops']['score_delta']:+.9f}", flush=True,
        )
    # Deployment is fitted only after the protocol and blend have been frozen by
    # the OOF experiment above.  This profile is never used to report OOF gains.
    deployment_profile = fit(combined, VF_FEATURES)
    serializable_profiles = {
        name: {
            str(fold): {
                **{k: v for k, v in profile.items() if k not in {"mean", "scale", "coefficients", "names"}},
                "names": list(profile["names"]),
                "mean": profile["mean"].tolist(),
                "scale": profile["scale"].tolist(),
                "coefficients": profile["coefficients"].tolist(),
                "intercept": profile["intercept"],
            }
            for fold, profile in fold_profiles.items()
        }
        for name, fold_profiles in profiles.items()
    }
    result = {
        "protocol": {"folds": FOLDS, "l2": L2, "match_floor": MATCH_FLOOR},
        "fold_audit": audit,
        "profiles": serializable_profiles,
        "deployment": {
            "variant": "A3b_vf_b75",
            "feature_set": "base_vf",
            "blend": 0.75,
            "profile": {
                **{
                    key: value
                    for key, value in deployment_profile.items()
                    if key not in {"mean", "scale", "coefficients", "names"}
                },
                "names": list(deployment_profile["names"]),
                "mean": deployment_profile["mean"].tolist(),
                "scale": deployment_profile["scale"].tolist(),
                "coefficients": deployment_profile["coefficients"].tolist(),
                "intercept": deployment_profile["intercept"],
            },
        },
        "results": results,
        "decision": {
            "selected": max(results, key=lambda name: results[name]["weighted_score"]),
            "release_allowed": any(
                row["weighted_delta"] >= 0.20
                and all(row["views"][view]["score_delta"] >= 0.10 for view in VIEW_WEIGHTS)
                and all(row["views"][view]["map50_delta"] >= 0.005 for view in VIEW_WEIGHTS)
                for row in results.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
