"""Cross-view, group-OOF SAFE-R2 reranking with shared coefficients.

This is a deliberately small follow-up to ``analyze_safe_r2_crossfit.py``.
Both original images and grid crops are training augmentations, but every
official duplicate group is excluded from both views in its held-out fold.
Two invariant model forms and two pre-registered score blends are evaluated;
no boxes are added, deleted, or geometrically modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import VIEW_WEIGHTS, load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import (
    EXPECTED_BASELINE,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FOLDS,
    L2,
    PERMITTED,
    coefficient_bounds,
    enrich_view,
    fuse_v7,
    load_groups,
)
from steel_defect.classes import CLASS_NAMES


VARIANTS = {
    "B0_baseline": None,
    "B1_shared_b25": {"normalization": "global", "blend": 0.25},
    "B2_shared_b50": {"normalization": "global", "blend": 0.50},
    "B3_classnorm_b25": {"normalization": "class", "blend": 0.25},
    "B4_classnorm_b50": {"normalization": "class", "blend": 0.50},
}
STRICT_GATE = {
    "weighted_delta": 0.20,
    "view_score_delta": 0.10,
    "view_map50_delta": 0.005,
    "maximum_recall_drop": 0.002,
}


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def duplicate_robust_weights(records: list[dict]) -> np.ndarray:
    """Balance view/class/label, then groups, then examples within each group."""
    weights = np.zeros(len(records), dtype=np.float64)
    strata: dict[tuple[str, int, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(records):
        strata[(record["view"], record["class_id"], record["label"])][
            record["group"]
        ].append(index)
    for groups in strata.values():
        group_mass = 1.0 / len(groups)
        for indices in groups.values():
            per_example = group_mass / len(indices)
            weights[indices] = per_example
    if not np.all(weights > 0.0):
        raise RuntimeError("zero cross-view sample weight")
    weights *= len(records) / float(np.sum(weights))
    return weights


def build_scaler(records: list[dict], normalization: str) -> dict:
    matrix = np.stack([record["features"] for record in records])
    if normalization == "global":
        mean = np.mean(matrix, axis=0)
        scale = np.maximum(np.std(matrix, axis=0), 1e-6)
        return {
            "mode": "global",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }
    if normalization != "class":
        raise ValueError(f"unknown normalization: {normalization}")
    per_class = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = np.asarray(
            [record["class_id"] == class_id for record in records], dtype=bool
        )
        if not np.any(selected):
            per_class[class_name] = {
                "mean": np.zeros(len(FEATURE_NAMES)).tolist(),
                "scale": np.ones(len(FEATURE_NAMES)).tolist(),
            }
            continue
        class_matrix = matrix[selected]
        per_class[class_name] = {
            "mean": np.mean(class_matrix, axis=0).tolist(),
            "scale": np.maximum(np.std(class_matrix, axis=0), 1e-6).tolist(),
        }
    return {"mode": "class", "per_class": per_class}


def standardized_features(record: dict, scaler: dict) -> np.ndarray:
    if scaler["mode"] == "global":
        mean = np.asarray(scaler["mean"], dtype=np.float64)
        scale = np.asarray(scaler["scale"], dtype=np.float64)
    else:
        class_name = CLASS_NAMES[record["class_id"]]
        item = scaler["per_class"][class_name]
        mean = np.asarray(item["mean"], dtype=np.float64)
        scale = np.asarray(item["scale"], dtype=np.float64)
    return (record["features"] - mean) / scale


def fit_shared_model(records: list[dict], normalization: str) -> dict:
    scaler = build_scaler(records, normalization)
    matrix = np.stack([standardized_features(record, scaler) for record in records])
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    views = np.asarray([record["view"] == "grid_crops" for record in records], dtype=np.int64)
    weights = duplicate_robust_weights(records)
    denominator = float(np.sum(weights))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercepts, coefficients = parameters[:2], parameters[2:]
        logits = intercepts[views] + matrix @ coefficients
        loss = float(np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits))) / denominator
        loss += 0.5 * L2 * float(np.dot(coefficients, coefficients))
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        growth = np.exp(logits[~positive])
        probabilities[~positive] = growth / (1.0 + growth)
        residual = weights * (probabilities - labels) / denominator
        gradient = np.r_[
            np.sum(residual[views == 0]),
            np.sum(residual[views == 1]),
            matrix.T @ residual + L2 * coefficients,
        ]
        return loss, gradient

    bounds = [(None, None), (None, None)] + coefficient_bounds(FEATURE_NAMES)[1:]
    result = minimize(
        objective,
        np.zeros(2 + len(FEATURE_NAMES), dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"cross-view fit failed: {result.message}")
    return {
        "schema": "safe_r2_crossview_v1",
        "normalization": normalization,
        "features": list(FEATURE_NAMES),
        "scaler": scaler,
        "intercepts": {
            "original": float(result.x[0]),
            "grid_crops": float(result.x[1]),
        },
        "coefficients": {
            name: float(result.x[2 + index])
            for index, name in enumerate(FEATURE_NAMES)
        },
        "fit": {
            "samples": len(records),
            "positives": int(np.sum(labels)),
            "groups": len({record["group"] for record in records}),
            "l2": L2,
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "weighting": "equal view/class/label strata; equal groups; equal examples within group",
        },
    }


def sigmoid(value: float) -> float:
    if value >= 0.0:
        decay = np.exp(-value)
        return float(1.0 / (1.0 + decay))
    growth = np.exp(value)
    return float(growth / (1.0 + growth))


def apply_model(records: list[dict], fold_models: dict[int, dict], view: str,
                blend: float) -> list[dict]:
    output = []
    for record in records:
        model = fold_models[record["fold"]]
        standardized = standardized_features(record, model["scaler"])
        coefficients = np.asarray(
            [model["coefficients"][name] for name in FEATURE_NAMES], dtype=np.float64
        )
        quality = sigmoid(float(model["intercepts"][view] + standardized @ coefficients))
        prediction = dict(record["prediction"])
        base = max(float(prediction["score"]), 1e-12)
        prediction["score"] = float(base ** (1.0 - blend) * quality ** blend)
        output.append(prediction)
    return output


def summarize(per_view: dict[str, dict], baseline: dict[str, dict]) -> dict:
    weighted = sum(VIEW_WEIGHTS[view] * per_view[view]["score"] for view in VIEW_WEIGHTS)
    base_weighted = sum(VIEW_WEIGHTS[view] * baseline[view]["score"] for view in VIEW_WEIGHTS)
    return {
        "weighted_score": weighted,
        "weighted_delta": weighted - base_weighted,
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


def gate(result: dict, baseline: dict[str, dict]) -> tuple[bool, list[str]]:
    failures = []
    if result["weighted_delta"] < STRICT_GATE["weighted_delta"]:
        failures.append("weighted_delta<0.20")
    for view in VIEW_WEIGHTS:
        row = result["views"][view]
        if row["score_delta"] < STRICT_GATE["view_score_delta"]:
            failures.append(f"{view}:score_delta<0.10")
        if row["map50_delta"] < STRICT_GATE["view_map50_delta"]:
            failures.append(f"{view}:map50_delta<0.005")
        if row["r_micro"] < baseline[view]["r_micro"] - STRICT_GATE["maximum_recall_drop"]:
            failures.append(f"{view}:recall_drop>0.002")
        if row["predictions"] != baseline[view]["predictions"]:
            failures.append(f"{view}:membership_changed")
    return not failures, failures


def report(results: dict, decision: dict) -> str:
    lines = [
        "# SAFE-R2 cross-view robust group OOF", "",
        "Shared coefficients were trained on both views with exact official-group "
        "exclusion. Grid crop multiplicity was neutralized by group-balanced weights. "
        "No prediction was added, removed, or geometrically changed.", "",
        "|Variant|View|Score|Delta|P|R|mAP50|TP|FP|FN|Count|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, item in results.items():
        for view in VIEW_WEIGHTS:
            row = item["views"][view]
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
        for variant, item in results.items():
            values = [item["views"][view]["per_class"][name]["ap50"] for name in PERMITTED]
            lines.append(f"|{variant}|" + "|".join(f"{value:.6f}" for value in values) + "|")
        lines.append("")
    lines.extend([
        "## Decision", "",
        f"Gate: `{decision['gate']}`.", "",
        f"Best OOF variant: `{decision['best_variant']}`.", "",
        "A frozen deployment profile is emitted only when the same strict gate used "
        "in the first SAFE-R2 study passes in both views.",
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
        raise FileExistsError(f"refusing to overwrite: {args.output}")

    groups = load_groups(args.splits)
    cells = {}
    baselines = {}
    baseline_metrics = {}
    enriched = {}
    for view in VIEW_WEIGHTS:
        print(f"LOAD {view}", flush=True)
        cell_a = load_cell(args.cells_root / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.cells_root / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if abs(metric["score"] - EXPECTED_BASELINE[view]) > 1e-9:
            raise RuntimeError(f"A0 mismatch for {view}: {metric['score']}")
        cells[view] = {"a": cell_a, "b": cell_b}
        baselines[view] = baseline
        baseline_metrics[view] = metric
        enriched[view] = enrich_view(cell_a, cell_b, baseline, groups, view)
        for record in enriched[view]:
            record["view"] = view
        print(f"READY {view}: {len(enriched[view])} anchors", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    fold_models: dict[str, dict[int, dict]] = {"global": {}, "class": {}}
    fold_audit = {}
    for fold in range(FOLDS):
        train = [
            record for view in VIEW_WEIGHTS for record in enriched[view]
            if record["fold"] != fold
        ]
        heldout = {
            record["group"] for view in VIEW_WEIGHTS for record in enriched[view]
            if record["fold"] == fold
        }
        if heldout.intersection({record["group"] for record in train}):
            raise RuntimeError("cross-view official-group leakage")
        for normalization in fold_models:
            print(f"FIT fold={fold} normalization={normalization} n={len(train)}", flush=True)
            fold_models[normalization][fold] = fit_shared_model(train, normalization)
        fold_audit[str(fold)] = {
            "training_anchors": len(train),
            "heldout_groups": len(heldout),
            "heldout_original": sum(record["fold"] == fold for record in enriched["original"]),
            "heldout_grid_crops": sum(record["fold"] == fold for record in enriched["grid_crops"]),
        }

    results = {}
    for variant, specification in VARIANTS.items():
        per_view = {}
        for view in VIEW_WEIGHTS:
            predictions = (
                baselines[view]
                if specification is None
                else apply_model(
                    enriched[view], fold_models[specification["normalization"]],
                    view, specification["blend"],
                )
            )
            if len(predictions) != len(baselines[view]):
                raise RuntimeError("SAFE-R2 changed candidate membership")
            per_view[view] = metrics50(
                predictions, cells[view]["a"]["ground_truth"], cells[view]["a"]["sizes"]
            )
        results[variant] = summarize(per_view, baseline_metrics)
        print(
            f"OOF {variant}: delta={results[variant]['weighted_delta']:+.6f} "
            f"original={results[variant]['views']['original']['score_delta']:+.6f} "
            f"grid={results[variant]['views']['grid_crops']['score_delta']:+.6f}",
            flush=True,
        )

    checks = {}
    for variant in VARIANTS:
        if variant == "B0_baseline":
            continue
        passed, failures = gate(results[variant], baseline_metrics)
        checks[variant] = {"passed": passed, "failures": failures}
    eligible = [variant for variant, item in checks.items() if item["passed"]]
    # steel_defect.safe_r2 currently has a single scaler.  Class-normalized
    # models remain valid OOF diagnostics but cannot be frozen without changing
    # the runtime contract; only global-normalized variants are deployable.
    deployable = [
        variant for variant in eligible
        if VARIANTS[variant]["normalization"] == "global"
    ]
    best = max(
        (variant for variant in VARIANTS if variant != "B0_baseline"),
        key=lambda variant: (
            min(results[variant]["views"][view]["score_delta"] for view in VIEW_WEIGHTS),
            results[variant]["weighted_delta"],
        ),
    )
    selected = max(deployable, key=lambda variant: results[variant]["weighted_score"]) if deployable else None
    decision = {
        "gate": "PASS" if deployable else "STOP",
        "best_variant": best,
        "selected_variant": selected,
        "eligible_variants": eligible,
        "safe_r2_compatible_eligible_variants": deployable,
        "checks": checks,
        "strict_gate": STRICT_GATE,
        "baseline_exact": {view: baseline_metrics[view]["score"] for view in VIEW_WEIGHTS},
        "fold_audit": fold_audit,
        "test_or_final_data_used": False,
    }
    write_json(args.output / "decision.json", decision)
    write_json(args.output / "oof_results.json", {
        "results": results,
        "fold_models": fold_models,
    })
    (args.output / "REPORT.md").write_text(report(results, decision), encoding="utf-8")

    if deployable:
        specification = VARIANTS[selected]
        all_records = [record for view in VIEW_WEIGHTS for record in enriched[view]]
        crossview = fit_shared_model(all_records, specification["normalization"])
        scaler = crossview["scaler"]
        frozen = {
            "support_iou": 0.60,
            "score_blend": specification["blend"],
            "scaler": {
                "mean": dict(zip(FEATURE_NAMES, scaler["mean"], strict=True)),
                "scale": dict(zip(FEATURE_NAMES, scaler["scale"], strict=True)),
            },
            "linear": {
                "intercept": crossview["intercepts"]["original"],
                "coefficients": crossview["coefficients"],
            },
            "refinement": {"enabled": False},
        }
        frozen["metadata"] = {
            "selected_variant": selected,
            "group_oof_gate": "PASS",
            "training_views": ["original", "grid_crops"],
            "training_group_policy": "official groups excluded from both views per heldout fold",
            "training_weighting": "equal view/class/label strata; equal groups; equal examples within group",
            "deployment_semantics": "original",
            "calibration_intercept_source": "original",
            "membership_preserving": True,
            "localization_changed": False,
            "test_or_final_labels_used": False,
            "crossview_fit": crossview["fit"],
        }
        write_json(args.output / "frozen_profile.json", frozen)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
