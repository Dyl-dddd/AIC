"""Low-capacity nonlinear SAFE-R2 candidate-quality evaluation.

This analyzer is intentionally isolated from the deployable SAFE-R2 module.
It trains deterministic, depth-two histogram gradient-boosted classifiers on
both original and grid-crop examples while holding out complete official
duplicate groups from both views.  The only operation at inference time is a
geometric score blend: membership and coordinates are immutable.

The model family, capacity, sparse-class fallback, and two blend strengths are
pre-registered constants rather than values selected on the OOF result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import VIEW_WEIGHTS, load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import (
    EXPECTED_BASELINE,
    FEATURE_NAMES,
    FOLDS,
    PERMITTED,
    enrich_view,
    fuse_v7,
    load_groups,
)
from steel_defect.classes import CLASS_NAMES


SEED = 20260928
SUPPORT_IOU = 0.60  # inherited by enrich_view; recorded here for the report
BLENDS = (0.25, 0.50)
MODEL_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.08,
    "max_iter": 24,
    "max_leaf_nodes": 4,
    "max_depth": 2,
    "min_samples_leaf": 80,
    "l2_regularization": 2.0,
    "max_bins": 32,
    "early_stopping": False,
    "random_state": SEED,
}
MIN_CLASS_POSITIVES = 24
MIN_CLASS_POSITIVE_GROUPS = 8
MIN_CLASS_NEGATIVE_GROUPS = 8
STRICT_GATE = {
    "weighted_delta": 0.20,
    "view_score_delta": 0.10,
    "view_map50_delta": 0.005,
    "maximum_recall_drop": 0.002,
}


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def lean_records(records: list[dict], view: str) -> list[dict]:
    """Discard geometry-only diagnostics after established feature extraction."""
    return [
        {
            "prediction": record["prediction"],
            "features": record["features"],
            "class_id": int(record["class_id"]),
            "label": int(record["label"]),
            "group": str(record["group"]),
            "fold": int(record["fold"]),
            "view": view,
        }
        for record in records
    ]


def duplicate_robust_weights(records: list[dict]) -> np.ndarray:
    """Equalize view/label strata, groups within strata, then rows in groups.

    The function is used separately for each class model.  Consequently each
    view/label combination gets equal total mass, an official duplicate group
    gets equal mass within that stratum, and extra sources/crops in one group
    cannot manufacture training weight.
    """
    buckets: dict[tuple[str, int], dict[str, list[int]]] = {}
    for index, record in enumerate(records):
        key = (record["view"], record["label"])
        buckets.setdefault(key, {}).setdefault(record["group"], []).append(index)
    required = {(view, label) for view in VIEW_WEIGHTS for label in (0, 1)}
    if set(buckets) != required:
        raise ValueError(f"incomplete view/label strata: {sorted(set(buckets))}")
    weights = np.zeros(len(records), dtype=np.float64)
    for groups in buckets.values():
        group_mass = 1.0 / len(groups)
        for indices in groups.values():
            row_mass = group_mass / len(indices)
            weights[indices] = row_mass
    if not np.all(weights > 0.0):
        raise RuntimeError("non-positive duplicate-robust training weight")
    weights *= len(weights) / float(np.sum(weights))
    return weights


def fit_classifier(records: list[dict]) -> HistGradientBoostingClassifier:
    matrix = np.stack([record["features"] for record in records]).astype(np.float32)
    labels = np.asarray([record["label"] for record in records], dtype=np.int8)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("classifier training requires both labels")
    weights = duplicate_robust_weights(records)
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    model.fit(matrix, labels, sample_weight=weights)
    return model


def class_is_supported(records: list[dict]) -> tuple[bool, dict]:
    positives = [record for record in records if record["label"] == 1]
    positive_groups = {record["group"] for record in positives}
    negative_groups = {record["group"] for record in records if record["label"] == 0}
    audit = {
        "rows": len(records),
        "positives": len(positives),
        "positive_groups": len(positive_groups),
        "negative_groups": len(negative_groups),
        "strata": {
            f"{view}:{label}": len({
                record["group"] for record in records
                if record["view"] == view and record["label"] == label
            })
            for view in VIEW_WEIGHTS for label in (0, 1)
        },
    }
    supported = (
        len(positives) >= MIN_CLASS_POSITIVES
        and len(positive_groups) >= MIN_CLASS_POSITIVE_GROUPS
        and len(negative_groups) >= MIN_CLASS_NEGATIVE_GROUPS
        and all(audit["strata"].values())
    )
    return supported, audit


def fit_fold_bundle(records: list[dict]) -> tuple[dict, dict]:
    """Fit one global fallback and supported per-class low-capacity models."""
    global_model = fit_classifier(records)
    models: dict[int, HistGradientBoostingClassifier] = {}
    audit = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        selected = [record for record in records if record["class_id"] == class_id]
        if not selected:
            continue
        supported, item = class_is_supported(selected)
        item["model"] = "per_class" if supported else "global_fallback"
        audit[class_name] = item
        if supported:
            models[class_id] = fit_classifier(selected)
    return {"global": global_model, "classes": models}, audit


def predict_oof_quality(records: list[dict], fold_models: dict[int, dict]) -> np.ndarray:
    quality = np.empty(len(records), dtype=np.float64)
    assigned = np.zeros(len(records), dtype=bool)
    by_fold_class: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(records):
        by_fold_class.setdefault((record["fold"], record["class_id"]), []).append(index)
    for (fold, class_id), indices in sorted(by_fold_class.items()):
        bundle = fold_models[fold]
        model = bundle["classes"].get(class_id, bundle["global"])
        matrix = np.stack([records[index]["features"] for index in indices]).astype(np.float32)
        probabilities = model.predict_proba(matrix)
        positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
        quality[indices] = probabilities[:, positive_column]
        assigned[indices] = True
    if not np.all(assigned) or not np.all(np.isfinite(quality)):
        raise RuntimeError("incomplete/non-finite OOF quality predictions")
    return np.clip(quality, 1e-12, 1.0)


def rerank(records: list[dict], quality: np.ndarray, blend: float) -> list[dict]:
    output = []
    for record, probability in zip(records, quality, strict=True):
        item = dict(record["prediction"])
        base = max(float(item["score"]), 1e-12)
        item["score"] = float(base ** (1.0 - blend) * float(probability) ** blend)
        output.append(item)
    return output


def membership_digest(predictions: list[dict]) -> str:
    """Order-independent multiset hash excluding only the mutable score."""
    rows = []
    for item in predictions:
        rows.append(json.dumps({
            "image_id": item["image_id"],
            "category_name": item["category_name"],
            "bbox": [float(value) for value in item["bbox"]],
        }, sort_keys=True, separators=(",", ":")))
    rows.sort()
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


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


def gate(result: dict, baseline: dict[str, dict]) -> tuple[bool, bool, list[str]]:
    directional = True
    failures = []
    if result["weighted_delta"] < STRICT_GATE["weighted_delta"]:
        failures.append("weighted_delta<0.20")
    for view in VIEW_WEIGHTS:
        row = result["views"][view]
        if row["score_delta"] <= 0.0 or row["map50_delta"] <= 0.0:
            directional = False
        if row["score_delta"] < STRICT_GATE["view_score_delta"]:
            failures.append(f"{view}:score_delta<0.10")
        if row["map50_delta"] < STRICT_GATE["view_map50_delta"]:
            failures.append(f"{view}:map50_delta<0.005")
        if row["r_micro"] < baseline[view]["r_micro"] - STRICT_GATE["maximum_recall_drop"]:
            failures.append(f"{view}:recall_drop>0.002")
        if row["predictions"] != baseline[view]["predictions"]:
            failures.append(f"{view}:membership_changed")
    return directional, not failures, failures


def render_report(results: dict, decision: dict, membership: dict, fold_audit: dict) -> str:
    lines = [
        "# SAFE-R2 low-capacity nonlinear group OOF", "",
        "A deterministic depth-two histogram gradient booster was trained on both "
        "views. Complete official duplicate groups were excluded from both views in "
        "each held-out fold. Training weights equalize view/label strata, official "
        "groups, and rows within a group. No box was added, removed, localized, or "
        "suppressed.", "",
        "|Variant|View|Score|Delta|P|R|mAP50|mAP delta|TP|FP|FN|Count|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, result in results.items():
        for view in VIEW_WEIGHTS:
            row = result["views"][view]
            lines.append(
                f"|{variant}|{view}|{row['score']:.6f}|{row['score_delta']:+.6f}|"
                f"{row['p_micro']:.7f}|{row['r_micro']:.7f}|{row['map50']:.7f}|"
                f"{row['map50_delta']:+.7f}|{row['tp']}|{row['fp']}|{row['fn']}|"
                f"{row['predictions']}|"
            )
    lines.extend(["", "## Per-class AP50", ""])
    for view in VIEW_WEIGHTS:
        lines.extend([
            f"### {view}", "",
            "|Variant|" + "|".join(PERMITTED) + "|",
            "|---|" + "|".join(["---:"] * len(PERMITTED)) + "|",
        ])
        for variant, result in results.items():
            values = [result["views"][view]["per_class"][name]["ap50"] for name in PERMITTED]
            lines.append(f"|{variant}|" + "|".join(f"{value:.6f}" for value in values) + "|")
        lines.append("")
    lines.extend([
        "## Membership invariants", "",
        "|View|Count|Baseline hash|All reranks identical|",
        "|---|---:|---|---|",
    ])
    for view in VIEW_WEIGHTS:
        item = membership[view]
        lines.append(
            f"|{view}|{item['count']}|`{item['baseline_sha256']}`|"
            f"{str(item['all_variants_identical'])}|"
        )
    lines.extend(["", "## Decision", ""])
    lines.append(f"Gate: `{decision['gate']}`. Deployment allowed: `{decision['deployment_allowed']}`.")
    lines.append("")
    lines.append(f"Selected diagnostic candidate: `{decision['selected_variant']}`.")
    lines.append("")
    lines.append(
        "A candidate is only deployable when weighted delta is at least +0.20, each "
        "view gains at least +0.10 score and +0.005 mAP50, recall loss is at most "
        "0.002, and membership is unchanged. A merely positive two-view result is "
        "reported as diagnostic evidence, not a deployable profile."
    )
    lines.extend(["", "## Capacity and runtime", ""])
    lines.append(
        f"scikit-learn `{decision['runtime']['sklearn_version']}`; fixed parameters "
        f"`{json.dumps(MODEL_PARAMS, sort_keys=True)}`."
    )
    lines.append("")
    lines.append(
        "The fitted estimator is not compatible with the current NumPy-only SAFE-R2 "
        "JSON profile. A passing model would require either the pinned scikit-learn "
        "runtime plus joblib artifacts, or a separately tested JSON tree exporter. "
        "No deployable artifact is emitted when the gate stops."
    )
    lines.extend(["", "## Fold audit", ""])
    lines.append("```json")
    lines.append(json.dumps(fold_audit, ensure_ascii=False, indent=2))
    lines.append("```")
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
    records: dict[str, list[dict]] = {}
    for view in VIEW_WEIGHTS:
        print(f"LOAD {view}", flush=True)
        cell_a = load_cell(args.cells_root / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.cells_root / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if abs(metric["score"] - EXPECTED_BASELINE[view]) > 1e-9:
            raise RuntimeError(
                f"baseline {view} mismatch: {metric['score']} != {EXPECTED_BASELINE[view]}"
            )
        print(f"BASELINE {view}: {metric['score']:.12f}", flush=True)
        enriched = enrich_view(cell_a, cell_b, baseline, groups, view)
        records[view] = lean_records(enriched, view)
        cells[view] = {"a": cell_a, "b": cell_b}
        baselines[view] = baseline
        baseline_metrics[view] = metric
        print(f"FEATURES {view}: {len(records[view])} rows", flush=True)

    fold_models = {}
    fold_audit = {}
    all_records = records["original"] + records["grid_crops"]
    for fold in range(FOLDS):
        train = [record for record in all_records if record["fold"] != fold]
        heldout = [record for record in all_records if record["fold"] == fold]
        train_groups = {record["group"] for record in train}
        heldout_groups = {record["group"] for record in heldout}
        overlap = train_groups.intersection(heldout_groups)
        if overlap:
            raise RuntimeError(f"official-group leakage in fold {fold}: {sorted(overlap)[:3]}")
        print(
            f"FIT fold={fold}: rows={len(train)} groups={len(train_groups)} "
            f"heldout_groups={len(heldout_groups)}",
            flush=True,
        )
        fold_models[fold], class_audit = fit_fold_bundle(train)
        fold_audit[str(fold)] = {
            "train_rows": len(train),
            "heldout_rows": len(heldout),
            "train_groups": len(train_groups),
            "heldout_groups": len(heldout_groups),
            "group_overlap": 0,
            "train_views": dict(Counter(record["view"] for record in train)),
            "heldout_views": dict(Counter(record["view"] for record in heldout)),
            "classes": class_audit,
        }

    qualities = {
        view: predict_oof_quality(records[view], fold_models)
        for view in VIEW_WEIGHTS
    }
    results = {
        "N0_baseline": summarize(baseline_metrics, baseline_metrics),
    }
    predictions_by_variant: dict[str, dict[str, list[dict]]] = {}
    for blend in BLENDS:
        variant = f"N1_hgb_depth2_b{int(round(100 * blend)):02d}"
        per_view = {}
        predictions_by_variant[variant] = {}
        for view in VIEW_WEIGHTS:
            predictions = rerank(records[view], qualities[view], blend)
            predictions_by_variant[variant][view] = predictions
            per_view[view] = metrics50(
                predictions, cells[view]["a"]["ground_truth"], cells[view]["a"]["sizes"]
            )
        results[variant] = summarize(per_view, baseline_metrics)
        print(
            f"OOF {variant}: weighted_delta={results[variant]['weighted_delta']:+.6f}",
            flush=True,
        )

    membership = {}
    for view in VIEW_WEIGHTS:
        baseline_hash = membership_digest(baselines[view])
        variant_hashes = {
            variant: membership_digest(predictions_by_variant[variant][view])
            for variant in predictions_by_variant
        }
        membership[view] = {
            "count": len(baselines[view]),
            "baseline_sha256": baseline_hash,
            "variant_sha256": variant_hashes,
            "all_variants_identical": all(value == baseline_hash for value in variant_hashes.values()),
        }
        if not membership[view]["all_variants_identical"]:
            raise RuntimeError(f"membership invariant failed for {view}")

    checks = {}
    candidates = [name for name in results if name != "N0_baseline"]
    for variant in candidates:
        directional, strict, failures = gate(results[variant], baseline_metrics)
        checks[variant] = {
            "directional_both_view_positive": directional,
            "strict_pass": strict,
            "failures": failures,
        }
    eligible = [name for name in candidates if checks[name]["strict_pass"]]
    selected = max(candidates, key=lambda name: results[name]["weighted_score"])
    decision = {
        "gate": "PASS" if eligible else "STOP",
        "deployment_allowed": bool(eligible),
        "selected_variant": (
            max(eligible, key=lambda name: results[name]["weighted_score"])
            if eligible else selected
        ),
        "selected_is_diagnostic_only": not bool(eligible),
        "eligible_variants": eligible,
        "gate_checks": checks,
        "protocol": {
            "folds": FOLDS,
            "fold_assignment": "sha256(official_group)[:8] modulo 5 (inherited)",
            "training_views": list(VIEW_WEIGHTS),
            "group_exclusion_applies_to_both_views": True,
            "support_iou": SUPPORT_IOU,
            "blends_pre_registered": list(BLENDS),
            "class_model_minimums": {
                "positives": MIN_CLASS_POSITIVES,
                "positive_groups": MIN_CLASS_POSITIVE_GROUPS,
                "negative_groups": MIN_CLASS_NEGATIVE_GROUPS,
            },
            "model_params": MODEL_PARAMS,
        },
        "runtime": {
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
            "numpy_only_safe_r2_compatible": False,
            "full_fit_artifact_emitted": False,
            "implication": "pin sklearn/joblib or implement and validate a JSON tree exporter",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "oof_results.json", {"results": results})
    write_json(args.output / "decision.json", decision)
    write_json(args.output / "membership.json", membership)
    write_json(args.output / "fold_audit.json", fold_audit)
    write_json(args.output / "model_spec.json", {
        "feature_names": list(FEATURE_NAMES),
        "model_params": MODEL_PARAMS,
        "blends": list(BLENDS),
        "seed": SEED,
        "note": "specification only; OOF fold estimators are deliberately not deployment artifacts",
    })
    (args.output / "REPORT.md").write_text(
        render_report(results, decision, membership, fold_audit), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
