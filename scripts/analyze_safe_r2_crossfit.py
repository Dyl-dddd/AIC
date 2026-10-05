"""Cross-fitted evaluation of SAFE-R2 on the frozen dual-view TTA dev cells.

Training uses original-view anchors only.  Every prediction is scored by a
model that did not see any source from its official duplicate group; the same
fold model is then replayed on the corresponding held-out grid crops.  AP is
computed once on the concatenated OOF predictions, never averaged by fold.
"""
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

from scripts.analyze_final_submission_postprocess import (
    VIEW_WEIGHTS,
    load_cell,
    metrics50,
    replay,
)
from scripts.analyze_source_aware_fusion import source_aware_fuse
from steel_defect.classes import CLASS_NAMES, CLASS_TO_ID
from steel_defect.geometry import box_iou
from steel_defect.safe_r2 import (
    FEATURE_NAMES,
    extract_features,
    guarded_robust_box_refinement,
    load_profile,
    quality_score,
    select_provenance_representatives,
)


EXPECTED_BASELINE = {
    "original": 68.7028528202815,
    "grid_crops": 69.1144764490038,
}
SUPPORT_IOU = 0.60
FOLDS = 5
L2 = 0.01
PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FEATURE_SETS = {
    "score_only": ("log_base_score",),
    "score_ab": ("log_base_score", "ab_presence", "ab_iou"),
    "full": FEATURE_NAMES,
}
VARIANTS = {
    "A0_baseline": {"feature_set": None, "blend": 0.0, "refine": False},
    "A1_score_only": {"feature_set": "score_only", "blend": 0.25, "refine": False},
    "A2_score_plus_ab": {"feature_set": "score_ab", "blend": 0.25, "refine": False},
    "A3_full_rerank": {"feature_set": "full", "blend": 0.25, "refine": False},
    "A3b_full_rerank_blend50": {"feature_set": "full", "blend": 0.50, "refine": False},
    "A4_ug_rbv_only": {"feature_set": None, "blend": 0.0, "refine": True},
    "A5_full_plus_ug_rbv": {"feature_set": "full", "blend": 0.25, "refine": True},
}
REFINEMENT = {
    "enabled": True,
    "max_dispersion": 0.20,
    "minimum_anchor_iou": 0.75,
    "minimum_donors": 2,
    "require_non_edge_local": True,
    "score_power": 0.5,
    "anchor_iou_power": 1.0,
}


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def source_key(image_id: str) -> str:
    return image_id.split("::", 1)[0]


def fold_for_group(group: str) -> int:
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def load_groups(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("splits records must be keyed by source filename")
    groups = {name: str(record["group"]) for name, record in records.items()}
    if not groups:
        raise ValueError("no official groups found")
    return groups


def fuse_v7(model_a: dict, model_b: dict) -> list[dict]:
    if model_a["ground_truth"] != model_b["ground_truth"]:
        raise ValueError("Model A/B ground truth mismatch")
    if model_a["sizes"] != model_b["sizes"]:
        raise ValueError("Model A/B image-size mismatch")
    return source_aware_fuse(
        replay(model_a), replay(model_b),
        match_iou=0.70,
        boost=1.10,
        coord_mode="anchor",
        unmatched_b_factor=0.50,
        b_scale=0.35,
        final_nms_iou=0.72,
    )


def label_anchor(prediction: dict, ground_truth: dict) -> int:
    boxes = ground_truth.get(prediction["category_name"], [])
    if not boxes:
        return 0
    overlaps = box_iou(
        np.asarray(prediction["bbox"], dtype=np.float32),
        np.asarray(boxes, dtype=np.float32),
    )
    return int(float(np.max(overlaps)) >= 0.50)


def enrich_view(
    cell_a: dict,
    cell_b: dict,
    baseline: list[dict],
    groups: dict[str, str],
    view: str,
) -> list[dict]:
    rows_a = {row["image_id"]: row for row in cell_a["rows"]}
    rows_b = {row["image_id"]: row for row in cell_b["rows"]}
    predictions: dict[str, list[dict]] = defaultdict(list)
    for item in baseline:
        predictions[item["image_id"]].append(item)
    output: list[dict] = []
    for image_number, image_id in enumerate(sorted(predictions)):
        row_a, row_b = rows_a[image_id], rows_b[image_id]
        source_id = str(row_a.get("source_id", source_key(image_id)))
        if source_id != str(row_b.get("source_id", source_key(image_id))):
            raise ValueError(f"source mismatch: {image_id}")
        if source_id not in groups:
            raise ValueError(f"missing official group: {source_id}")
        by_class_a: dict[int, list[dict]] = defaultdict(list)
        by_class_b: dict[int, list[dict]] = defaultdict(list)
        for item in row_a["candidates"]:
            by_class_a[int(item["class_id"])].append(item)
        for item in row_b["candidates"]:
            by_class_b[int(item["class_id"])].append(item)
        for prediction in predictions[image_id]:
            class_id = CLASS_TO_ID[prediction["category_name"]]
            representatives = select_provenance_representatives(
                prediction,
                by_class_a[class_id],
                by_class_b[class_id],
                SUPPORT_IOU,
            )
            features = extract_features(
                prediction, representatives, tuple(row_a["image_size"])
            )
            output.append({
                "prediction": prediction,
                "features": np.asarray(
                    [features[name] for name in FEATURE_NAMES], dtype=np.float64
                ),
                "representatives": representatives,
                "image_size": tuple(row_a["image_size"]),
                "class_id": class_id,
                "label": label_anchor(prediction, row_a["ground_truth"]),
                "group": groups[source_id],
                "fold": fold_for_group(groups[source_id]),
            })
        if (image_number + 1) % 100 == 0:
            print(
                f"FEATURES {view}: {image_number + 1}/{len(predictions)} images, "
                f"{len(output)} anchors",
                flush=True,
            )
    return output


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
    if not strata or not np.all(weights > 0.0):
        raise ValueError("cannot construct complete class/label balanced weights")
    weights *= len(weights) / float(np.sum(weights))
    return weights


def coefficient_bounds(names: tuple[str, ...]) -> list[tuple[float | None, float | None]]:
    nonnegative = {
        "log_base_score", "ab_presence", "ab_iou", "source_count",
        "global_local_coherence",
    }
    nonpositive = {"dispersion", "edge_only"}
    bounds: list[tuple[float | None, float | None]] = [(None, None)]
    for name in names:
        if name in nonnegative:
            bounds.append((0.0, None))
        elif name in nonpositive:
            bounds.append((None, 0.0))
        else:
            bounds.append((None, None))
    return bounds


def fit_profile(records: list[dict], names: tuple[str, ...]) -> dict:
    indices = np.asarray([FEATURE_INDEX[name] for name in names], dtype=np.int64)
    matrix = np.stack([record["features"][indices] for record in records])
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    classes = np.asarray([record["class_id"] for record in records], dtype=np.int64)
    weights = balanced_weights(classes, labels)
    means = np.mean(matrix, axis=0)
    scales = np.std(matrix, axis=0)
    scales = np.maximum(scales, 1e-6)
    standardized = (matrix - means) / scales
    denominator = float(np.sum(weights))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coefficients = parameters[0], parameters[1:]
        logits = intercept + standardized @ coefficients
        loss = float(np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits))) / denominator
        loss += 0.5 * L2 * float(np.dot(coefficients, coefficients))
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        growth = np.exp(logits[~positive])
        probabilities[~positive] = growth / (1.0 + growth)
        residual = weights * (probabilities - labels) / denominator
        gradient = np.r_[np.sum(residual), standardized.T @ residual + L2 * coefficients]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(len(names) + 1, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=coefficient_bounds(names),
        options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    coefficients = {name: 0.0 for name in FEATURE_NAMES}
    mean_map = {name: 0.0 for name in FEATURE_NAMES}
    scale_map = {name: 1.0 for name in FEATURE_NAMES}
    for index, name in enumerate(names):
        coefficients[name] = float(result.x[index + 1])
        mean_map[name] = float(means[index])
        scale_map[name] = float(scales[index])
    return load_profile({
        "support_iou": SUPPORT_IOU,
        "score_blend": 0.0,
        "scaler": {"mean": mean_map, "scale": scale_map},
        "linear": {
            "intercept": float(result.x[0]),
            "coefficients": coefficients,
        },
        "refinement": REFINEMENT,
        "fit": {
            "features": list(names),
            "l2": L2,
            "samples": len(records),
            "positives": int(np.sum(labels)),
            "objective": float(result.fun),
            "iterations": int(result.nit),
        },
    })


def apply_records(
    records: list[dict],
    fold_profiles: dict[int, dict[str, dict]],
    variant: str,
) -> tuple[list[dict], dict]:
    specification = VARIANTS[variant]
    output = []
    refined = 0
    reasons: dict[str, int] = defaultdict(int)
    for record in records:
        prediction = dict(record["prediction"])
        profile = None
        if specification["feature_set"] is not None:
            profile = fold_profiles[record["fold"]][specification["feature_set"]]
            features = {
                name: float(record["features"][index])
                for index, name in enumerate(FEATURE_NAMES)
            }
            quality = quality_score(features, profile)
            base = max(float(prediction["score"]), 1e-12)
            blend = float(specification["blend"])
            prediction["score"] = float(base ** (1.0 - blend) * quality ** blend)
        if specification["refine"]:
            refine_profile = profile or load_profile({
                "support_iou": SUPPORT_IOU,
                "refinement": REFINEMENT,
            })
            box, changed, reason = guarded_robust_box_refinement(
                record["prediction"], record["representatives"],
                record["image_size"], refine_profile,
            )
            prediction["bbox"] = box
            refined += int(changed)
            reasons[reason] += 1
        output.append(prediction)
    return output, {"refined": refined, "refinement_reasons": dict(sorted(reasons.items()))}


def summarize_variant(per_view: dict[str, dict], baseline: dict[str, dict]) -> dict:
    weighted = sum(VIEW_WEIGHTS[view] * per_view[view]["score"] for view in VIEW_WEIGHTS)
    baseline_weighted = sum(
        VIEW_WEIGHTS[view] * baseline[view]["score"] for view in VIEW_WEIGHTS
    )
    return {
        "weighted_score": weighted,
        "weighted_delta": weighted - baseline_weighted,
        "views": {
            view: {
                **per_view[view],
                "score_delta": per_view[view]["score"] - baseline[view]["score"],
                "map50_delta": per_view[view]["map50"] - baseline[view]["map50"],
                "recall_delta": per_view[view]["r_micro"] - baseline[view]["r_micro"],
            }
            for view in VIEW_WEIGHTS
        },
    }


def passes_gate(result: dict, baseline: dict[str, dict]) -> tuple[bool, list[str]]:
    failures = []
    if result["weighted_delta"] < 0.20:
        failures.append("weighted_delta<0.20")
    for view in VIEW_WEIGHTS:
        row = result["views"][view]
        if row["score_delta"] < 0.10:
            failures.append(f"{view}:score_delta<0.10")
        if row["map50_delta"] < 0.005:
            failures.append(f"{view}:map50_delta<0.005")
        if row["r_micro"] < baseline[view]["r_micro"] - 0.002:
            failures.append(f"{view}:recall_drop>0.002")
        if row["predictions"] > baseline[view]["predictions"]:
            failures.append(f"{view}:membership_increased")
    return not failures, failures


def render_report(results: dict, decision: dict) -> str:
    lines = [
        "# SAFE-R2 group cross-fitting", "",
        "All reranking metrics are concatenated 5-fold OOF results. Training used only "
        "the original view; each fold model was also replayed on grid crops from the same "
        "held-out official groups.", "",
        "|Variant|View|Score|Delta|P|R|mAP50|TP|FP|FN|Predictions|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in results.items():
        for view in VIEW_WEIGHTS:
            row = result["views"][view]
            lines.append(
                f"|{variant}|{view}|{row['score']:.6f}|{row['score_delta']:+.6f}|"
                f"{row['p_micro']:.7f}|{row['r_micro']:.7f}|{row['map50']:.7f}|"
                f"{row['tp']}|{row['fp']}|{row['fn']}|{row['predictions']}|"
            )
    lines.extend(["", "## Per-class AP50", ""])
    for view in VIEW_WEIGHTS:
        lines.extend([
            f"### {view}", "",
            "|Variant|" + "|".join(PERMITTED) + "|",
            "|---|" + "|".join(["---:"] * len(PERMITTED)) + "|",
        ])
        for variant, result in results.items():
            aps = [result["views"][view]["per_class"][name]["ap50"] for name in PERMITTED]
            lines.append(f"|{variant}|" + "|".join(f"{value:.6f}" for value in aps) + "|")
        lines.append("")
    lines.extend([
        "## Decision", "",
        f"Gate: `{decision['gate']}`.", "",
        f"Selected diagnostic candidate: `{decision['selected_variant']}`.", "",
        "Strict release thresholds: weighted delta >= 0.20, both view score deltas "
        ">= 0.10, both view mAP50 deltas >= 0.005, recall drop <= 0.002, and no "
        "membership increase.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells-root", type=Path,
        default=ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells",
    )
    parser.add_argument(
        "--splits", type=Path,
        default=ROOT / "data/official_v2_20260831b/splits.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    groups = load_groups(args.splits)

    cells: dict[str, dict[str, dict]] = {}
    baselines: dict[str, list[dict]] = {}
    baseline_metrics: dict[str, dict] = {}
    for view in VIEW_WEIGHTS:
        print(f"LOAD {view}", flush=True)
        cell_a = load_cell(args.cells_root / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.cells_root / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if abs(metric["score"] - EXPECTED_BASELINE[view]) > 1e-9:
            raise RuntimeError(
                f"A0 {view} mismatch: {metric['score']} != {EXPECTED_BASELINE[view]}"
            )
        print(f"A0 {view}: {metric['score']:.12f} ({len(baseline)} predictions)", flush=True)
        cells[view] = {"a": cell_a, "b": cell_b}
        baselines[view] = baseline
        baseline_metrics[view] = metric

    args.output.mkdir(parents=True, exist_ok=False)
    enriched: dict[str, list[dict]] = {}
    for view in VIEW_WEIGHTS:
        enriched[view] = enrich_view(
            cells[view]["a"], cells[view]["b"], baselines[view], groups, view
        )
        print(f"FEATURES {view}: complete ({len(enriched[view])} anchors)", flush=True)

    fold_profiles: dict[int, dict[str, dict]] = {}
    fold_audit = {}
    for fold in range(FOLDS):
        train = [record for record in enriched["original"] if record["fold"] != fold]
        heldout_groups = sorted({
            record["group"] for record in enriched["original"] if record["fold"] == fold
        })
        train_groups = {record["group"] for record in train}
        if train_groups.intersection(heldout_groups):
            raise RuntimeError("group leakage detected")
        fold_profiles[fold] = {}
        for feature_set, names in FEATURE_SETS.items():
            print(f"FIT fold={fold} features={feature_set} samples={len(train)}", flush=True)
            fold_profiles[fold][feature_set] = fit_profile(train, names)
        fold_audit[str(fold)] = {
            "train_anchors": len(train),
            "heldout_original_anchors": sum(
                record["fold"] == fold for record in enriched["original"]
            ),
            "heldout_grid_anchors": sum(
                record["fold"] == fold for record in enriched["grid_crops"]
            ),
            "heldout_groups": len(heldout_groups),
        }

    results = {}
    variant_diagnostics = {}
    for variant in VARIANTS:
        per_view = {}
        variant_diagnostics[variant] = {}
        for view in VIEW_WEIGHTS:
            if variant == "A0_baseline":
                predictions = baselines[view]
                diagnostic = {"refined": 0, "refinement_reasons": {}}
            else:
                predictions, diagnostic = apply_records(
                    enriched[view], fold_profiles, variant
                )
            per_view[view] = metrics50(
                predictions, cells[view]["a"]["ground_truth"], cells[view]["a"]["sizes"]
            )
            variant_diagnostics[variant][view] = diagnostic
        results[variant] = summarize_variant(per_view, baseline_metrics)
        print(
            f"OOF {variant}: weighted={results[variant]['weighted_score']:.6f} "
            f"delta={results[variant]['weighted_delta']:+.6f}",
            flush=True,
        )

    gate_checks = {}
    for variant, result in results.items():
        if variant == "A0_baseline":
            continue
        passed, failures = passes_gate(result, baseline_metrics)
        gate_checks[variant] = {"passed": passed, "failures": failures}
    eligible = [variant for variant, check in gate_checks.items() if check["passed"]]
    ranking = sorted(
        (variant for variant in results if variant != "A0_baseline"),
        key=lambda variant: (
            results[variant]["weighted_score"],
            min(results[variant]["views"][view]["score"] for view in VIEW_WEIGHTS),
            variant,
        ),
        reverse=True,
    )
    selected = max(eligible, key=lambda variant: results[variant]["weighted_score"]) if eligible else ranking[0]
    decision = {
        "gate": "PASS" if eligible else "STOP",
        "selected_variant": selected,
        "deployment_allowed": bool(eligible),
        "eligible_variants": eligible,
        "gate_checks": gate_checks,
        "baseline_exact": {
            view: baseline_metrics[view]["score"] for view in VIEW_WEIGHTS
        },
        "protocol": {
            "folds": FOLDS,
            "fold_assignment": "sha256(official_group)[:8] modulo 5",
            "training_view": "original only",
            "support_iou": SUPPORT_IOU,
            "l2": L2,
            "fold_audit": fold_audit,
        },
    }

    specification = VARIANTS[selected]
    feature_set = specification["feature_set"] or "full"
    fitted = fit_profile(enriched["original"], FEATURE_SETS[feature_set])
    fitted["score_blend"] = float(specification["blend"])
    fitted["refinement"]["enabled"] = bool(specification["refine"])
    fitted["metadata"] = {
        "method": "SAFE-R2",
        "selected_from_group_oof": selected,
        "deployment_allowed": bool(eligible),
        "gate": decision["gate"],
        "training_view": "original",
        "test_labels_used": False,
    }
    payload = {
        "results": results,
        "variant_diagnostics": variant_diagnostics,
        "fold_profiles": fold_profiles,
    }
    write_json(args.output / "oof_results.json", payload)
    write_json(args.output / "decision.json", decision)
    write_json(args.output / "fitted_profile.json", fitted)
    (args.output / "REPORT.md").write_text(
        render_report(results, decision), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
