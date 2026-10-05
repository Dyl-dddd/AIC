"""Evaluate one fixed, unsupervised, view-invariant CAGE-Q revision.

CAGE-Q-VI intentionally discards provenance availability, edge flags, and
local/global counts because those variables have different acquisition
semantics in original images and grid crops.  It retains only cross-model box
agreement, whose meaning is shared by both views, and maps that agreement to a
per-class empirical percentile without labels.  There is one preregistered
amplitude and no threshold sweep, deletion, NMS, or localization change.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import VIEW_WEIGHTS, load_cell, metrics50, replay
from scripts.analyze_safe_r2_heuristic import _representatives_for_group
from scripts.analyze_source_aware_fusion import source_aware_fuse
from steel_defect.safe_r2 import extract_features


CELLS = ROOT / (
    "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/"
    "ensemble_v6_tta_dev/cells"
)
OUTPUT = ROOT / "runs/semifinal/safe_r2_view_invariant_20260928"
EXPECTED = {
    "original": 68.7028528202815,
    "grid_crops": 69.1144764490038,
}
AMPLITUDE = 0.08
AGREEMENT_FLOOR = 0.60


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fuse_v7(cell_a: dict, cell_b: dict) -> list[dict]:
    return source_aware_fuse(
        replay(cell_a), replay(cell_b),
        match_iou=0.70,
        boost=1.10,
        coord_mode="anchor",
        unmatched_b_factor=0.50,
        b_scale=0.35,
        final_nms_iou=0.72,
    )


def average_percentiles(values: np.ndarray) -> np.ndarray:
    """Average-rank empirical percentiles in (0, 1), deterministic for ties."""
    if not len(values):
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return (ranks + 0.5) / len(values)


def cage_q_vi(baseline: list[dict], cell_a: dict, cell_b: dict) -> tuple[list[dict], dict]:
    """Apply the single CAGE-Q-VI ranking rule while preserving membership."""
    rows_a = {row["image_id"]: row for row in cell_a["rows"]}
    rows_b = {row["image_id"]: row for row in cell_b["rows"]}
    grouped: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for index, item in enumerate(baseline):
        grouped[(item["image_id"], item["category_name"])].append((index, item))

    strength = np.zeros(len(baseline), dtype=np.float64)
    supported = np.zeros(len(baseline), dtype=bool)
    for group_number, ((image_id, _), indexed) in enumerate(grouped.items(), 1):
        indices = [pair[0] for pair in indexed]
        predictions = [pair[1] for pair in indexed]
        representatives = _representatives_for_group(
            predictions, rows_a[image_id], rows_b[image_id]
        )
        size = tuple(cell_a["sizes"][image_id])
        for index, prediction, reps in zip(indices, predictions, representatives, strict=True):
            features = extract_features(prediction, reps, size)
            if features["ab_presence"] > 0.0 and features["ab_iou"] > AGREEMENT_FLOOR:
                supported[index] = True
                strength[index] = float(
                    np.clip(
                        (features["ab_iou"] - AGREEMENT_FLOOR)
                        / (1.0 - AGREEMENT_FLOOR),
                        0.0,
                        1.0,
                    )
                )
        if group_number % 2000 == 0:
            print(f"  evidence {group_number}/{len(grouped)} image/class groups", flush=True)

    delta = np.zeros(len(baseline), dtype=np.float64)
    class_diagnostics = {}
    classes = np.asarray([item["category_name"] for item in baseline], dtype=object)
    for class_name in sorted(set(classes.tolist())):
        indices = np.flatnonzero((classes == class_name) & supported)
        if len(indices):
            percentile = average_percentiles(strength[indices])
            # Strength keeps near-floor agreements near neutral; the percentile
            # residual removes view-specific availability and scale inflation.
            delta[indices] = (
                AMPLITUDE * strength[indices] * (2.0 * percentile - 1.0)
            )
        class_diagnostics[class_name] = {
            "predictions": int(np.sum(classes == class_name)),
            "supported": int(len(indices)),
            "support_fraction": float(len(indices) / max(np.sum(classes == class_name), 1)),
            "mean_strength": float(np.mean(strength[indices])) if len(indices) else 0.0,
            "mean_delta": float(np.mean(delta[indices])) if len(indices) else 0.0,
        }

    output = []
    raw_maximum = 0.0
    for item, adjustment in zip(baseline, delta, strict=True):
        revised = dict(item)
        revised["score"] = float(item["score"]) * math.exp(float(adjustment))
        raw_maximum = max(raw_maximum, revised["score"])
        output.append(revised)
    divisor = max(1.0, raw_maximum)
    if divisor > 1.0:
        for item in output:
            item["score"] /= divisor
    if len(output) != len(baseline):
        raise AssertionError("CAGE-Q-VI changed membership")
    diagnostics = {
        "formula": (
            "delta=0.08*strength*(2*class_ecdf(strength)-1); "
            "strength=clip((AB_iou-0.60)/0.40,0,1); unsupported delta=0"
        ),
        "amplitude": AMPLITUDE,
        "agreement_floor": AGREEMENT_FLOOR,
        "prediction_count": len(output),
        "supported": int(np.sum(supported)),
        "delta": {
            "minimum": float(np.min(delta)),
            "maximum": float(np.max(delta)),
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "positive": int(np.sum(delta > 0.0)),
            "negative": int(np.sum(delta < 0.0)),
            "zero": int(np.sum(delta == 0.0)),
        },
        "score_legalization_divisor": divisor,
        "per_class": class_diagnostics,
    }
    return output, diagnostics


def metric_delta(result: dict, baseline: dict) -> dict:
    return {
        "score": result["score"] - baseline["score"],
        "p_micro": result["p_micro"] - baseline["p_micro"],
        "r_micro": result["r_micro"] - baseline["r_micro"],
        "map50": result["map50"] - baseline["map50"],
        "tp": result["tp"] - baseline["tp"],
        "fp": result["fp"] - baseline["fp"],
        "fn": result["fn"] - baseline["fn"],
        "per_class_ap50": {
            name: result["per_class"][name]["ap50"] - baseline["per_class"][name]["ap50"]
            for name in baseline["per_class"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")

    baseline_metrics = {}
    revised_metrics = {}
    diagnostics = {}
    for view in VIEW_WEIGHTS:
        print(f"REPLAY {view}", flush=True)
        cell_a = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(CELLS / f"model_b_tta_s1280_{view}")
        if cell_a["ground_truth"] != cell_b["ground_truth"] or cell_a["sizes"] != cell_b["sizes"]:
            raise ValueError(f"unaligned cells: {view}")
        baseline = fuse_v7(cell_a, cell_b)
        baseline_result = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if not math.isclose(baseline_result["score"], EXPECTED[view], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"baseline mismatch {view}: {baseline_result['score']} != {EXPECTED[view]}"
            )
        print(f"  baseline={baseline_result['score']:.15f}", flush=True)
        revised, diagnostics[view] = cage_q_vi(baseline, cell_a, cell_b)
        if any(left["bbox"] != right["bbox"] for left, right in zip(baseline, revised, strict=True)):
            raise AssertionError("CAGE-Q-VI changed localization")
        revised_result = metrics50(revised, cell_a["ground_truth"], cell_a["sizes"])
        revised_result["delta_vs_baseline"] = metric_delta(revised_result, baseline_result)
        baseline_metrics[view] = baseline_result
        revised_metrics[view] = revised_result
        print(
            f"  cage_q_vi={revised_result['score']:.15f} "
            f"delta={revised_result['score'] - baseline_result['score']:+.6f}",
            flush=True,
        )

    baseline_weighted = sum(
        VIEW_WEIGHTS[view] * baseline_metrics[view]["score"] for view in VIEW_WEIGHTS
    )
    revised_weighted = sum(
        VIEW_WEIGHTS[view] * revised_metrics[view]["score"] for view in VIEW_WEIGHTS
    )
    directional = all(
        revised_metrics[view]["score"] > baseline_metrics[view]["score"]
        for view in VIEW_WEIGHTS
    )
    strict = (
        revised_weighted - baseline_weighted >= 0.20
        and all(
            revised_metrics[view]["score"] - baseline_metrics[view]["score"] >= 0.10
            and revised_metrics[view]["map50"] - baseline_metrics[view]["map50"] >= 0.005
            for view in VIEW_WEIGHTS
        )
    )
    decision = {
        "study": "single preregistered unsupervised CAGE-Q-VI revision",
        "baseline_exact": {view: baseline_metrics[view]["score"] for view in VIEW_WEIGHTS},
        "protocol": {
            "parameter_sweep": False,
            "labels_used_for_ranking": False,
            "membership_changes": False,
            "localization_changes": False,
            "second_nms": False,
            "same_formula_both_views": True,
            "amplitude": AMPLITUDE,
            "agreement_floor": AGREEMENT_FLOOR,
        },
        "baseline_weighted": baseline_weighted,
        "cage_q_vi_weighted": revised_weighted,
        "weighted_delta": revised_weighted - baseline_weighted,
        "directional_both_view_pass": directional,
        "strict_release_pass": strict,
        "gate": "PASS_STRICT" if strict else "STOP",
        "views": {
            view: {
                "baseline": baseline_metrics[view],
                "cage_q_vi": revised_metrics[view],
            }
            for view in VIEW_WEIGHTS
        },
        "diagnostics": diagnostics,
        "domain_shift": {
            "original_single_source_fraction": 89649 / 133664,
            "grid_single_source_fraction": 34405 / 144294,
            "original_four_source_fraction": 6190 / 133664,
            "grid_four_source_fraction": 29127 / 144294,
            "original_edge_penalized_predictions": 14233,
            "grid_edge_penalized_predictions": 0,
            "interpretation": (
                "source availability and edge flags are acquisition-dependent, so CAGE-Q-VI "
                "removes them and uses only class-normalized cross-model agreement"
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "decision.json", decision)

    lines = [
        "# SAFE-R2 view-invariant CAGE-Q evaluation",
        "",
        "One fixed unsupervised revision was tested. There was no threshold sweep, label fitting, deletion, NMS, or localization change.",
        "",
        "## Domain shift diagnosis",
        "",
        "|Signal|Original|Grid crops|",
        "|---|---:|---:|",
        f"|Single-source boxes|{89649 / 133664:.1%}|{34405 / 144294:.1%}|",
        f"|Four-source boxes|{6190 / 133664:.1%}|{29127 / 144294:.1%}|",
        "|Edge-penalized boxes|14,233|0|",
        "",
        "The original rule mixed quality with acquisition-dependent source availability. CAGE-Q-VI retains only cross-model IoU and converts it to an unlabeled, per-class empirical-percentile residual.",
        "",
        "## Raw results",
        "",
        "|System|View|Score|Delta|P|R|mAP50|TP|FP|FN|Predictions|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in VIEW_WEIGHTS:
        for tag, result in (("V7 primary", baseline_metrics[view]), ("CAGE-Q-VI", revised_metrics[view])):
            delta = result["score"] - baseline_metrics[view]["score"]
            lines.append(
                f"|{tag}|{view}|{result['score']:.6f}|{delta:+.6f}|"
                f"{result['p_micro']:.7f}|{result['r_micro']:.7f}|{result['map50']:.7f}|"
                f"{result['tp']}|{result['fp']}|{result['fn']}|{result['predictions']}|"
            )
    lines.extend([
        "",
        f"Weighted baseline: `{baseline_weighted:.6f}`. Weighted CAGE-Q-VI: `{revised_weighted:.6f}` (delta `{revised_weighted - baseline_weighted:+.6f}`).",
        "",
        "## Per-class AP50 and deltas",
        "",
        "|Class|Original baseline|Original VI|Delta|Grid baseline|Grid VI|Delta|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in baseline_metrics["original"]["per_class"]:
        ob = baseline_metrics["original"]["per_class"][name]["ap50"]
        ov = revised_metrics["original"]["per_class"][name]["ap50"]
        gb = baseline_metrics["grid_crops"]["per_class"][name]["ap50"]
        gv = revised_metrics["grid_crops"]["per_class"][name]["ap50"]
        lines.append(f"|{name}|{ob:.6f}|{ov:.6f}|{ov-ob:+.6f}|{gb:.6f}|{gv:.6f}|{gv-gb:+.6f}|")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Directional both-view pass: `{directional}`.",
        f"Strict release pass: `{strict}`.",
        f"Gate: `{decision['gate']}`.",
        "",
        "The strict gate requires weighted delta >= 0.20, and each view score delta >= 0.10 and mAP50 delta >= 0.005.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "weighted_delta": decision["weighted_delta"],
        "directional_both_view_pass": directional,
        "strict_release_pass": strict,
        "gate": decision["gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
