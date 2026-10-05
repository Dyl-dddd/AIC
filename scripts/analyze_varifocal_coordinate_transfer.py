"""Evaluate frozen Varifocal score transfer plus bounded coordinate interpolation."""

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

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_varifocal_oof_transfer import (
    VIEW_WEIGHTS, apply, enrich, load_groups,
)
from scripts.analyze_varifocal_score_transfer import iou_matrix
from steel_defect.classes import CLASS_NAMES
from steel_defect.metrics import _match_class, average_precision


EXPECTED_V7 = {"original": 68.70285282028146, "grid_crops": 69.11447644900377}
VARIANTS = {
    "C0_rescore_only": None,
    "C1_iou70_a25": (0.70, 0.25),
    "C2_iou70_a50": (0.70, 0.50),
    "C3_iou85_a25": (0.85, 0.25),
    "C4_iou85_a50": (0.85, 0.50),
}


def restore_profile(payload: dict) -> dict:
    return {
        **payload,
        "names": tuple(payload["names"]),
        "mean": np.asarray(payload["mean"], dtype=np.float64),
        "scale": np.asarray(payload["scale"], dtype=np.float64),
        "coefficients": np.asarray(payload["coefficients"], dtype=np.float64),
    }


def coordinate_matches(baseline: list[dict], quality: list[dict]) -> list[tuple[float, list[float] | None]]:
    base_groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    quality_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for index, item in enumerate(baseline):
        base_groups[(item["image_id"], item["category_name"])].append((index, item))
    for item in quality:
        quality_groups[(item["image_id"], item["category_name"])].append(item)
    matches: list[tuple[float, list[float] | None]] = [(0.0, None) for _ in baseline]
    for key, rows in base_groups.items():
        candidates = quality_groups.get(key, [])
        if not candidates:
            continue
        left = np.asarray([item[1]["bbox"] for item in rows], dtype=np.float32)
        right = np.asarray([item["bbox"] for item in candidates], dtype=np.float32)
        matrix = iou_matrix(left, right)
        best = matrix.argmax(axis=1)
        for local, quality_index in enumerate(best):
            matches[rows[local][0]] = (
                float(matrix[local, int(quality_index)]),
                [float(value) for value in candidates[int(quality_index)]["bbox"]],
            )
    return matches


def interpolate(
    predictions: list[dict], matches: list[tuple[float, list[float] | None]],
    sizes: dict, floor: float, alpha: float,
) -> tuple[list[dict], int]:
    output, changed = [], 0
    for item, (overlap, quality_box) in zip(predictions, matches, strict=True):
        row = dict(item)
        if quality_box is not None and overlap >= floor:
            base = np.asarray(item["bbox"], dtype=np.float64)
            quality = np.asarray(quality_box, dtype=np.float64)
            box = (1.0 - alpha) * base + alpha * quality
            width, height = sizes[item["image_id"]]
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            rounded = [int(round(value)) for value in box]
            if rounded[2] > rounded[0] and rounded[3] > rounded[1]:
                row["bbox"] = rounded
                changed += int(rounded != item["bbox"])
        output.append(row)
    return output, changed


def summarize(predictions: list[dict], cell: dict) -> dict:
    metric = metrics50(predictions, cell["ground_truth"], cell["sizes"])
    ap75 = []
    for name in (name for name in CLASS_NAMES if name != "qilie"):
        tp, fp, count = _match_class(predictions, cell["ground_truth"], name, 0.75)
        if count:
            ap75.append(average_precision(tp, fp, count))
    return {
        **metric,
        "map75": float(np.mean(ap75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--oof-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = load_groups(args.splits)
    calibration = json.loads(args.oof_profile.read_text(encoding="utf-8"))
    profiles = {
        int(fold): restore_profile(profile)
        for fold, profile in calibration["profiles"]["base_vf"].items()
    }
    cells, rescored, matches = {}, {}, {}
    for view in VIEW_WEIGHTS:
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        metric = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if not math.isclose(metric["score"], EXPECTED_V7[view], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"V7 mismatch: {view}")
        quality = json.loads((
            args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json"
        ).read_text(encoding="utf-8"))
        records = enrich(baseline, quality, cell_a["ground_truth"], groups, view)
        cells[view] = cell_a
        rescored[view] = apply(records, profiles, 0.75)
        matches[view] = coordinate_matches(baseline, quality)
        print(f"READY {view}: {len(baseline)}", flush=True)

    results = {}
    base_metrics = {view: summarize(rescored[view], cells[view]) for view in VIEW_WEIGHTS}
    base_weighted = sum(VIEW_WEIGHTS[view] * base_metrics[view]["score"] for view in VIEW_WEIGHTS)
    for name, config in VARIANTS.items():
        views, changed = {}, {}
        for view in VIEW_WEIGHTS:
            predictions = rescored[view]
            changed[view] = 0
            if config is not None:
                predictions, changed[view] = interpolate(
                    predictions, matches[view], cells[view]["sizes"], *config
                )
            metric = summarize(predictions, cells[view])
            views[view] = {
                **metric,
                "score_delta": metric["score"] - base_metrics[view]["score"],
                "map50_delta": metric["map50"] - base_metrics[view]["map50"],
                "map75_delta": metric["map75"] - base_metrics[view]["map75"],
            }
        weighted = sum(VIEW_WEIGHTS[view] * views[view]["score"] for view in VIEW_WEIGHTS)
        results[name] = {
            "config": config,
            "weighted_score": weighted,
            "weighted_delta_vs_rescore": weighted - base_weighted,
            "changed_boxes": changed,
            "views": views,
        }
        print(
            f"{name}: delta={weighted-base_weighted:+.9f} "
            f"original={views['original']['score_delta']:+.9f} "
            f"grid={views['grid_crops']['score_delta']:+.9f} "
            f"ap75=({views['original']['map75_delta']:+.6f},"
            f"{views['grid_crops']['map75_delta']:+.6f})", flush=True,
        )
    selected = max(results, key=lambda name: results[name]["weighted_score"])
    payload = {
        "baseline": "Varifocal OOF B75 score transfer",
        "variants": VARIANTS,
        "results": results,
        "selected": selected,
        "release_allowed": (
            results[selected]["weighted_delta_vs_rescore"] >= 0.35
            and all(results[selected]["views"][view]["score_delta"] >= 0.10 for view in VIEW_WEIGHTS)
            and all(results[selected]["views"][view]["map75_delta"] >= 0.008 for view in VIEW_WEIGHTS)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
