"""Evaluate pre-registered, learning-free SAFE-R2 controls on frozen TTA dev.

The study is deliberately small: an exact V7-primary replay is the negative
control, CAGE-Q is the positive ranking intervention, and UG-RBV is evaluated
alone and in combination with CAGE-Q.  Prediction membership is immutable and
no second NMS pass is allowed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import (
    VIEW_WEIGHTS,
    load_cell,
    metrics50,
    replay,
)
from scripts.analyze_source_aware_fusion import source_aware_fuse
from steel_defect.classes import CLASS_NAMES
from steel_defect.safe_r2 import (
    extract_features,
    guarded_robust_box_refinement,
)


EXPECTED_BASELINE = {
    "original": 68.7028528202815,
    "grid_crops": 69.1144764490038,
}
SUPPORT_IOU = 0.60
BASE = Path(
    "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/"
    "ensemble_v6_tta_dev/cells"
)
DEFAULT_OUTPUT = Path("runs/semifinal/safe_r2_heuristic_20260927")
VIEWS = {
    "original": (
        BASE / "model_a_tta_s1024_original",
        BASE / "model_b_tta_s1280_original",
    ),
    "grid_crops": (
        BASE / "model_a_tta_s1024_grid_crops",
        BASE / "model_b_tta_s1280_grid_crops",
    ),
}
REFINEMENT_PROFILE = {
    "refinement": {
        "enabled": True,
        "max_dispersion": 0.20,
        "minimum_anchor_iou": 0.85,
        "minimum_donors": 2,
        "require_non_edge_local": True,
        "score_power": 0.5,
        "anchor_iou_power": 1.0,
    }
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return dense IoUs for one image/class/bucket, never across the dataset."""
    if not len(left) or not len(right):
        return np.empty((len(left), len(right)), dtype=np.float64)
    left = left.astype(np.float64, copy=False)
    right = right.astype(np.float64, copy=False)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.maximum(0.0, bottom_right - top_left)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(
        0.0, left[:, 3] - left[:, 1]
    )
    right_area = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(
        0.0, right[:, 3] - right[:, 1]
    )
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def _raw_buckets(
    row_a: Mapping, row_b: Mapping, class_name: str
) -> dict[str, list[dict]]:
    """Index raw candidates into the four capped provenance buckets."""
    output: dict[str, list[dict]] = defaultdict(list)
    for model, row in (("a", row_a), ("b", row_b)):
        for candidate in row["candidates"]:
            candidate_class = CLASS_NAMES[int(candidate["class_id"])]
            if candidate_class != class_name:
                continue
            view = "global" if bool(candidate.get("is_global", False)) else "local"
            output[f"{model}_{view}"].append(candidate)
    return output


def _representatives_for_group(
    predictions: Sequence[Mapping], row_a: Mapping, row_b: Mapping
) -> list[dict[str, dict]]:
    """Vectorized anchor reconnect with one representative per A/B x L/G bucket."""
    if not predictions:
        return []
    class_name = str(predictions[0]["category_name"])
    anchors = np.asarray([item["bbox"] for item in predictions], dtype=np.float64)
    selected: list[dict[str, dict]] = [dict() for _ in predictions]
    for bucket, candidates in _raw_buckets(row_a, row_b, class_name).items():
        if not candidates:
            continue
        boxes = np.asarray([item["bbox"] for item in candidates], dtype=np.float64)
        scores = np.asarray([float(item["score"]) for item in candidates])
        overlaps = _pairwise_iou(anchors, boxes)
        for anchor_index, row_overlaps in enumerate(overlaps):
            best_iou = float(np.max(row_overlaps))
            if best_iou < SUPPORT_IOU:
                continue
            tied = np.flatnonzero(row_overlaps == best_iou)
            if len(tied) > 1:
                best_score = float(np.max(scores[tied]))
                tied = tied[scores[tied] == best_score]
            candidate_index = int(tied[0])
            representative = dict(candidates[candidate_index])
            representative["anchor_iou"] = best_iou
            representative["model"] = bucket[0]
            representative["provenance_bucket"] = bucket
            selected[anchor_index][bucket] = representative
    return selected


def _cage_delta(features: Mapping[str, float], representatives: Mapping) -> float:
    """Pre-registered CAGE-Q evidence delta; no labels or fitted parameters."""
    ab = float(features["ab_presence"])
    ab_iou = float(features["ab_iou"])
    coherence = []
    for model in ("a", "b"):
        local = representatives.get(f"{model}_local")
        global_ = representatives.get(f"{model}_global")
        if local is not None and global_ is not None:
            values = _pairwise_iou(
                np.asarray([local["bbox"]], dtype=np.float64),
                np.asarray([global_["bbox"]], dtype=np.float64),
            )
            coherence.append(float(values[0, 0]))
    lg = float(bool(coherence))
    lg_iou = float(np.mean(coherence)) if coherence else 0.0
    source_count = min(4.0, max(1.0, float(features["source_count"])))
    delta = (
        0.18 * ab * np.clip((ab_iou - 0.60) / 0.40, 0.0, 1.0)
        + 0.08 * lg * np.clip((lg_iou - 0.60) / 0.40, 0.0, 1.0)
        + 0.05 * math.log2(source_count) / 2.0
        - 0.10 * float(features["edge_only"])
    )
    return float(np.clip(delta, -0.10, 0.25))


def _legalize_proportional_scores(predictions: list[dict]) -> float:
    """Apply one global scale if exp(delta) exceeds the submission score range.

    A common positive multiplier preserves every CAGE-Q ordering exactly and is
    used only to satisfy the repository's [0, 1] score schema.
    """
    maximum = max((float(item["score"]) for item in predictions), default=1.0)
    divisor = max(1.0, maximum)
    if divisor > 1.0:
        for item in predictions:
            item["score"] = float(item["score"]) / divisor
    return divisor


def apply_controls(
    baseline: list[dict], cell_a: dict, cell_b: dict
) -> tuple[dict[str, list[dict]], dict]:
    """Produce CAGE-Q/refinement controls without changing membership."""
    by_image_class: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for index, prediction in enumerate(baseline):
        by_image_class[(prediction["image_id"], prediction["category_name"])].append(
            (index, prediction)
        )
    rows_a = {row["image_id"]: row for row in cell_a["rows"]}
    rows_b = {row["image_id"]: row for row in cell_b["rows"]}
    cage = [dict(item) for item in baseline]
    refine = [dict(item) for item in baseline]
    combined = [dict(item) for item in baseline]
    deltas = np.zeros(len(baseline), dtype=np.float64)
    source_counts = Counter()
    bucket_counts = Counter()
    refinement_reasons = Counter()
    refined_count = 0

    for group_index, ((image_id, _), indexed) in enumerate(by_image_class.items(), 1):
        indices = [item[0] for item in indexed]
        predictions = [item[1] for item in indexed]
        representatives = _representatives_for_group(
            predictions, rows_a[image_id], rows_b[image_id]
        )
        image_size = tuple(cell_a["sizes"][image_id])
        for index, prediction, reps in zip(indices, predictions, representatives, strict=True):
            features = extract_features(prediction, reps, image_size)
            delta = _cage_delta(features, reps)
            deltas[index] = delta
            source_counts[int(features["source_count"])] += 1
            bucket_counts.update(reps.keys())
            cage[index]["score"] = float(prediction["score"]) * math.exp(delta)
            combined[index]["score"] = cage[index]["score"]
            box, changed, reason = guarded_robust_box_refinement(
                prediction, reps, image_size, REFINEMENT_PROFILE
            )
            refinement_reasons[reason] += 1
            if changed:
                refined_count += 1
                refine[index]["bbox"] = box
                combined[index]["bbox"] = box
        if group_index % 1000 == 0:
            print(f"  reconnected {group_index}/{len(by_image_class)} image/class groups", flush=True)

    cage_divisor = _legalize_proportional_scores(cage)
    combined_divisor = _legalize_proportional_scores(combined)
    if len(cage) != len(baseline) or len(refine) != len(baseline) or len(combined) != len(baseline):
        raise AssertionError("SAFE-R2 controls must preserve prediction membership")
    diagnostics = {
        "support_iou": SUPPORT_IOU,
        "prediction_count": len(baseline),
        "delta": {
            "minimum": float(np.min(deltas)),
            "maximum": float(np.max(deltas)),
            "mean": float(np.mean(deltas)),
            "median": float(np.median(deltas)),
            "positive": int(np.sum(deltas > 0.0)),
            "negative": int(np.sum(deltas < 0.0)),
            "zero": int(np.sum(deltas == 0.0)),
        },
        "source_count_histogram": dict(sorted(source_counts.items())),
        "bucket_support_counts": dict(sorted(bucket_counts.items())),
        "refined_count": refined_count,
        "refined_fraction": refined_count / max(len(baseline), 1),
        "refinement_reasons": dict(sorted(refinement_reasons.items())),
        "score_legalization_divisor": {
            "cage_q": cage_divisor,
            "cage_q_plus_refine": combined_divisor,
        },
    }
    return {
        "cage_q": cage,
        "ug_rbv_refine_only": refine,
        "cage_q_plus_ug_rbv": combined,
    }, diagnostics


def _delta_metrics(metrics: dict, baseline: dict) -> dict:
    output = {
        key: metrics[key] - baseline[key]
        for key in ("score", "p_micro", "r_micro", "map50", "tp", "fp", "fn")
    }
    output["per_class_ap50"] = {
        name: metrics["per_class"][name]["ap50"] - values["ap50"]
        for name, values in baseline["per_class"].items()
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing study: {args.output}")

    loaded = {}
    baselines = {}
    baseline_metrics = {}
    for view, (path_a, path_b) in VIEWS.items():
        print(f"Loading and replaying V7 primary: {view}", flush=True)
        cell_a = load_cell(path_a)
        cell_b = load_cell(path_b)
        if cell_a["ground_truth"] != cell_b["ground_truth"] or cell_a["sizes"] != cell_b["sizes"]:
            raise ValueError(f"unaligned TTA cells: {view}")
        baseline = source_aware_fuse(
            replay(cell_a), replay(cell_b),
            match_iou=0.70,
            boost=1.10,
            coord_mode="anchor",
            unmatched_b_factor=0.50,
            b_scale=0.35,
            final_nms_iou=0.72,
        )
        result = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        print(f"  baseline {view}: {result['score']:.15f}", flush=True)
        if not math.isclose(result["score"], EXPECTED_BASELINE[view], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"baseline mismatch for {view}: {result['score']!r} != "
                f"{EXPECTED_BASELINE[view]!r}; stopping before interventions"
            )
        loaded[view] = (cell_a, cell_b)
        baselines[view] = baseline
        baseline_metrics[view] = result

    records: dict[str, dict[str, dict]] = defaultdict(dict)
    diagnostics = {}
    for view in VIEWS:
        print(f"Running frozen SAFE-R2 controls: {view}", flush=True)
        cell_a, cell_b = loaded[view]
        variants, diagnostics[view] = apply_controls(baselines[view], cell_a, cell_b)
        baseline_result = dict(baseline_metrics[view])
        baseline_result["delta_vs_baseline"] = _delta_metrics(
            baseline_metrics[view], baseline_metrics[view]
        )
        records["v7_primary_baseline"][view] = baseline_result
        for tag, predictions in variants.items():
            result = metrics50(predictions, cell_a["ground_truth"], cell_a["sizes"])
            result["delta_vs_baseline"] = _delta_metrics(result, baseline_metrics[view])
            records[tag][view] = result
            print(
                f"  {tag} {view}: {result['score']:.15f} "
                f"(delta {result['score'] - baseline_metrics[view]['score']:+.6f})",
                flush=True,
            )

    methods = []
    for tag, views in records.items():
        weighted = sum(VIEW_WEIGHTS[view] * views[view]["score"] for view in VIEW_WEIGHTS)
        baseline_weighted = sum(
            VIEW_WEIGHTS[view] * baseline_metrics[view]["score"] for view in VIEW_WEIGHTS
        )
        methods.append({
            "tag": tag,
            "weighted_score": weighted,
            "delta_weighted": weighted - baseline_weighted,
            "beats_baseline_both_views": all(
                views[view]["score"] > baseline_metrics[view]["score"]
                for view in VIEW_WEIGHTS
            ) if tag != "v7_primary_baseline" else False,
            "views": views,
        })
    methods.sort(key=lambda item: item["weighted_score"], reverse=True)
    eligible = [item for item in methods if item["beats_baseline_both_views"]]
    decision = {
        "study": "pre-registered learning-free CAGE-Q and UG-RBV controls",
        "negative_control": "v7_primary_baseline",
        "positive_control": "cage_q",
        "constraints": {
            "support_iou": SUPPORT_IOU,
            "provenance_bucket_cap": 4,
            "membership_changes": False,
            "second_nms": False,
            "same_parameters_both_views": True,
            "refinement": REFINEMENT_PROFILE["refinement"],
        },
        "baseline_reproduced": True,
        "diagnostics": diagnostics,
        "methods": methods,
        "recommended": eligible[0] if eligible else None,
        "gate": "IMPLEMENT" if eligible else "REJECT_HEURISTIC_SAFE_R2",
        "warning": "Frozen development evidence only; no Test labels were used.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / "decision.json", decision)

    lines = [
        "# SAFE-R2 heuristic preregistered evaluation",
        "",
        "No learning, threshold fitting, deletion, or second NMS was used. The same fixed parameters were replayed on both frozen views.",
        "",
        "## Raw results",
        "",
        "|Method|Original|Grid crops|Weighted|Delta weighted|P original/grid|R original/grid|mAP50 original/grid|TP original/grid|FP original/grid|FN original/grid|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in methods:
        original = item["views"]["original"]
        grid = item["views"]["grid_crops"]
        lines.append(
            f"|{item['tag']}|{original['score']:.6f}|{grid['score']:.6f}|"
            f"{item['weighted_score']:.6f}|{item['delta_weighted']:+.6f}|"
            f"{original['p_micro']:.6f}/{grid['p_micro']:.6f}|"
            f"{original['r_micro']:.6f}/{grid['r_micro']:.6f}|"
            f"{original['map50']:.6f}/{grid['map50']:.6f}|"
            f"{original['tp']}/{grid['tp']}|{original['fp']}/{grid['fp']}|"
            f"{original['fn']}/{grid['fn']}|"
        )
    lines.extend(["", "## Per-class AP50 deltas versus V7 primary", ""])
    for item in methods:
        if item["tag"] == "v7_primary_baseline":
            continue
        lines.extend([
            f"### {item['tag']}",
            "",
            "|Class|Original delta|Grid crops delta|",
            "|---|---:|---:|",
        ])
        for name in baseline_metrics["original"]["per_class"]:
            original_delta = item["views"]["original"]["delta_vs_baseline"]["per_class_ap50"][name]
            grid_delta = item["views"]["grid_crops"]["delta_vs_baseline"]["per_class_ap50"][name]
            lines.append(f"|{name}|{original_delta:+.6f}|{grid_delta:+.6f}|")
        lines.append("")
    lines.extend([
        "## Decision",
        "",
        f"Gate: `{decision['gate']}`.",
        f"Recommended: `{decision['recommended']['tag'] if decision['recommended'] else 'none'}`.",
        "",
        "Observation: CAGE-Q changes ranking only; therefore its TP/FP/recall must exactly match the baseline.",
        "Interpretation: any CAGE-Q score delta is attributable to evidence ordering, while UG-RBV deltas isolate localization.",
        "Implication: only a method improving both frozen views is eligible for implementation.",
        "Next step: package only the eligible fixed mechanism for a cache-level Test replay.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Decision: {decision['gate']}", flush=True)


if __name__ == "__main__":
    main()
