"""Run a bounded source-aware fusion study on returned native-TTA dev caches."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import (
    VIEW_WEIGHTS,
    load_cell,
    metrics50,
    replay,
    summarize,
    write_json,
)
from scripts.analyze_source_aware_fusion import source_aware_fuse


# The broad non-TTA study already rejected larger boosts and unmatched-B weight.
# This follow-up changes only match IoU and coordinate mode around that result.
CONFIGS = tuple(
    {
        "tag": f"tta_consensus_iou{match_iou:.2f}_{coordinate_mode}",
        "match_iou": match_iou,
        "boost": 1.10,
        "coordinate_mode": coordinate_mode,
        "unmatched_b_factor": 0.50,
        "model_b_score_scale": 0.35,
        "final_nms_iou": 0.72,
    }
    for match_iou in (0.50, 0.60, 0.70)
    for coordinate_mode in ("anchor", "vote")
)

OLD_V6 = {
    "weighted": 68.36661302454482,
    "original": 68.41827922265865,
    "grid_crops": 68.28256934227966,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-original", type=Path, required=True)
    parser.add_argument("--model-a-grid", type=Path, required=True)
    parser.add_argument("--model-b-original", type=Path, required=True)
    parser.add_argument("--model-b-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    paths = {
        "original": (args.model_a_original, args.model_b_original),
        "grid_crops": (args.model_a_grid, args.model_b_grid),
    }
    records: dict[str, dict[str, dict]] = defaultdict(dict)
    single_models: dict[str, dict[str, dict]] = defaultdict(dict)
    for view, (a_path, b_path) in paths.items():
        cell_a = load_cell(a_path)
        cell_b = load_cell(b_path)
        if cell_a["ground_truth"] != cell_b["ground_truth"] or cell_a["sizes"] != cell_b["sizes"]:
            raise ValueError(f"unaligned Model A/B TTA cells: {view}")
        model_a = replay(cell_a)
        model_b = replay(cell_b)
        single_models["model_a_tta"][view] = metrics50(
            model_a, cell_a["ground_truth"], cell_a["sizes"]
        )
        single_models["model_b_tta"][view] = metrics50(
            model_b, cell_a["ground_truth"], cell_a["sizes"]
        )
        for config in CONFIGS:
            predictions = source_aware_fuse(
                model_a,
                model_b,
                match_iou=config["match_iou"],
                boost=config["boost"],
                coord_mode=config["coordinate_mode"],
                unmatched_b_factor=config["unmatched_b_factor"],
                b_scale=config["model_b_score_scale"],
                final_nms_iou=config["final_nms_iou"],
            )
            result = metrics50(predictions, cell_a["ground_truth"], cell_a["sizes"])
            records[config["tag"]][view] = result
            print(f"{view} {config['tag']}: {result['score']:.6f}", flush=True)

    grid = summarize(records)
    for item in grid:
        item["delta_vs_old_v6"] = item["weighted_score"] - OLD_V6["weighted"]
        item["beats_old_v6_both_views"] = all(
            item["views"][view]["score"] >= OLD_V6[view]
            for view in VIEW_WEIGHTS
        )
    robust = [item for item in grid if item["beats_old_v6_both_views"]]
    recommendation = robust[0] if robust else None
    decision = {
        "old_v6": OLD_V6,
        "single_models": {
            name: {
                "weighted_score": sum(
                    VIEW_WEIGHTS[view] * values[view]["score"] for view in VIEW_WEIGHTS
                ),
                "views": values,
            }
            for name, values in single_models.items()
        },
        "best": grid[0],
        "robust_candidates": robust,
        "recommended": recommendation,
        "gate": "BUILD_TTA_TEST_PACKAGE" if recommendation else "KEEP_NON_TTA_V6",
        "warning": "Frozen-development proxy only; official 66.96 remains protected.",
    }
    write_json(args.output / "grid.json", grid)
    write_json(args.output / "decision.json", decision)

    lines = [
        "# Native-TTA source-aware fusion", "",
        "All values are frozen-development proxies under `20P + 60R + 20mAP50`.", "",
        "|System|Original|Grid crops|Weighted|Delta vs old V6|TP original/grid|FP original/grid|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in grid:
        lines.append(
            f"|{item['tag']}|{item['views']['original']['score']:.3f}|"
            f"{item['views']['grid_crops']['score']:.3f}|{item['weighted_score']:.3f}|"
            f"{item['delta_vs_old_v6']:+.3f}|{item['views']['original']['tp']}/"
            f"{item['views']['grid_crops']['tp']}|{item['views']['original']['fp']}/"
            f"{item['views']['grid_crops']['fp']}|"
        )
    lines.extend([
        "",
        f"Robust candidates beating old V6 in both views: `{len(robust)}`.",
        f"Decision gate: `{decision['gate']}`.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
