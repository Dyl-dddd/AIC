"""OOF AP-aligned rescoring with one-to-one GT labels and duplicate competition."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_varifocal_oof_transfer import (
    FOLDS, VIEW_WEIGHTS, VF_FEATURES, apply, fit, fold_for_group, load_groups,
    quality_matches, source_key,
)
from scripts.analyze_varifocal_score_transfer import iou_matrix
from steel_defect.classes import CLASS_TO_ID


EXPECTED_V7 = {"original": 68.70285282028146, "grid_crops": 69.11447644900377}
DUP_FEATURES = VF_FEATURES + (
    "log_peer_count_50", "max_peer_iou", "higher_peer_count_50",
    "is_cluster_top", "relative_class_rank",
)
FEATURE_SETS = {"unique_vf": VF_FEATURES, "unique_vf_dup": DUP_FEATURES}
VARIANTS = {
    "U0_v7": (None, 0.0),
    "U1_unique_vf_b75": ("unique_vf", 0.75),
    "U2_unique_dup_b50": ("unique_vf_dup", 0.50),
    "U3_unique_dup_b75": ("unique_vf_dup", 0.75),
    "U4_unique_dup_b100": ("unique_vf_dup", 1.00),
}


def unique_labels(baseline: list[dict], ground_truth: dict) -> list[int]:
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for index, prediction in enumerate(baseline):
        groups[(prediction["image_id"], prediction["category_name"])].append((index, prediction))
    labels = [0] * len(baseline)
    for (image_id, class_name), predictions in groups.items():
        truth = ground_truth[image_id].get(class_name, [])
        if not truth:
            continue
        left = np.asarray([item[1]["bbox"] for item in predictions], dtype=np.float32)
        right = np.asarray(truth, dtype=np.float32)
        overlaps = iou_matrix(left, right)
        rows, columns = linear_sum_assignment(-overlaps)
        for row, column in zip(rows, columns, strict=True):
            if float(overlaps[row, column]) >= 0.50:
                labels[predictions[int(row)][0]] = 1
    return labels


def duplicate_features(baseline: list[dict]) -> list[dict[str, float]]:
    groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for index, prediction in enumerate(baseline):
        groups[(prediction["image_id"], prediction["category_name"])].append((index, prediction))
    output = [{} for _ in baseline]
    for predictions in groups.values():
        boxes = np.asarray([item[1]["bbox"] for item in predictions], dtype=np.float32)
        scores = np.asarray([float(item[1]["score"]) for item in predictions], dtype=np.float64)
        overlaps = iou_matrix(boxes, boxes)
        np.fill_diagonal(overlaps, 0.0)
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[order] = np.arange(len(scores))
        for local, (global_index, _) in enumerate(predictions):
            peers = overlaps[local] >= 0.50
            higher = peers & (scores > scores[local])
            output[global_index] = {
                "log_peer_count_50": math.log1p(int(peers.sum())),
                "max_peer_iou": float(overlaps[local].max(initial=0.0)),
                "higher_peer_count_50": float(higher.sum()),
                "is_cluster_top": float(not higher.any()),
                "relative_class_rank": float(ranks[local]) / max(len(scores) - 1, 1),
            }
    return output


def enrich_view(
    baseline: list[dict], quality: list[dict], ground_truth: dict,
    official_groups: dict[str, str], view: str,
) -> list[dict]:
    matches = quality_matches(baseline, quality)
    labels = unique_labels(baseline, ground_truth)
    duplicates = duplicate_features(baseline)
    records = []
    for prediction, match, label, duplicate in zip(
        baseline, matches, labels, duplicates, strict=True
    ):
        base = max(float(prediction["score"]), 1e-12)
        vf = max(float(match["vf_score"]), 1e-12)
        features = {
            "log_base_score": math.log(base),
            "vf_present": float(match["vf_present"]),
            "vf_iou": float(match["vf_iou"]),
            "log_vf_score": math.log(vf),
            "vf_log_advantage": math.log(vf) - math.log(base),
            **duplicate,
        }
        group = official_groups[source_key(prediction["image_id"])]
        records.append({
            "prediction": prediction,
            "features": features,
            "label": label,
            "class_id": CLASS_TO_ID[prediction["category_name"]],
            "group": group,
            "fold": fold_for_group(group),
            "view": view,
        })
    return records


def serializable(profile: dict) -> dict:
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
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official_groups = load_groups(args.splits)
    cells, baselines, records = {}, {}, {}
    for view in VIEW_WEIGHTS:
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if not math.isclose(metric["score"], EXPECTED_V7[view], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"V7 baseline mismatch: {view}")
        quality = json.loads((
            args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json"
        ).read_text(encoding="utf-8"))
        cells[view], baselines[view] = cell_a, baseline
        records[view] = enrich_view(
            baseline, quality, cell_a["ground_truth"], official_groups, view
        )
        print(
            f"ENRICH {view}: rows={len(records[view])} "
            f"unique_positive={sum(row['label'] for row in records[view])}", flush=True,
        )
    combined = [*records["original"], *records["grid_crops"]]
    profiles: dict[str, dict[int, dict]] = {name: {} for name in FEATURE_SETS}
    fold_audit = {}
    for fold in range(FOLDS):
        train = [row for row in combined if row["fold"] != fold]
        heldout = {row["group"] for row in combined if row["fold"] == fold}
        if heldout & {row["group"] for row in train}:
            raise RuntimeError("group leakage")
        fold_audit[str(fold)] = {"train": len(train), "heldout_groups": len(heldout)}
        for name, feature_names in FEATURE_SETS.items():
            profiles[name][fold] = fit(train, feature_names)
            print(f"FIT fold={fold} set={name}", flush=True)

    baseline_metrics = {
        view: metrics50(baselines[view], cells[view]["ground_truth"], cells[view]["sizes"])
        for view in VIEW_WEIGHTS
    }
    baseline_weighted = sum(
        VIEW_WEIGHTS[view] * baseline_metrics[view]["score"] for view in VIEW_WEIGHTS
    )
    results = {}
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
        weighted = sum(VIEW_WEIGHTS[view] * views[view]["score"] for view in VIEW_WEIGHTS)
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
    selected = max(results, key=lambda name: results[name]["weighted_score"])
    selected_set, selected_blend = VARIANTS[selected]
    deployment = None if selected_set is None else {
        "variant": selected,
        "feature_set": selected_set,
        "blend": selected_blend,
        "profile": serializable(fit(combined, FEATURE_SETS[selected_set])),
    }
    payload = {
        "protocol": {
            "folds": FOLDS,
            "label": "maximum-IoU one-to-one Hungarian match at IoU>=0.50",
            "variants": VARIANTS,
        },
        "fold_audit": fold_audit,
        "results": results,
        "selected": selected,
        "deployment": deployment,
        "release_allowed": (
            results[selected]["weighted_delta"] >= 0.35
            and all(results[selected]["views"][view]["score_delta"] >= 0.10 for view in VIEW_WEIGHTS)
            and all(results[selected]["views"][view]["map50_delta"] >= 0.005 for view in VIEW_WEIGHTS)
        ),
        "profiles": {
            name: {str(fold): serializable(profile) for fold, profile in fold_profiles.items()}
            for name, fold_profiles in profiles.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
