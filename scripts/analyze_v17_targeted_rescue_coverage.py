"""Measure whether V17 recovers defects outside the protected V7/V12 universe."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_v13_expert_coverage import coverage, distribution


ROOT = Path(__file__).resolve().parents[1]
V7 = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
V17 = ROOT / "runs/v17_targeted_rescue_results_received_20261002/runs/semifinal/v17_targeted_rescue_pipeline/selection/cells"
OUTPUT = ROOT / "runs/semifinal/v17_targeted_rescue_analysis_20261002/coverage.json"


def main() -> None:
    result = {"views": {}}
    for view in ("original", "grid_crops"):
        a = load_cell(V7 / f"model_a_tta_s1024_{view}")
        b = load_cell(V7 / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(a, b)
        baseline_hits, _ = coverage(baseline, a["ground_truth"])
        folder = next(V17.glob(f"v17_targeted_rescue_last_*_{view}"))
        predictions = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
        hits, details = coverage(predictions, a["ground_truth"])
        unique = hits - baseline_hits
        lost = baseline_hits - hits
        result["views"][view] = {
            "baseline_metrics": metrics50(baseline, a["ground_truth"], a["sizes"]),
            "v17_metrics": metrics50(predictions, a["ground_truth"], a["sizes"]),
            "baseline_coverage_hits": len(baseline_hits),
            "v17_coverage_hits": len(hits),
            "unique_vs_v7_v12": len(unique),
            "v7_v12_only": len(lost),
            "union_hits": len(hits | baseline_hits),
            "unique_by_class": dict(sorted(Counter(key[1] for key in unique).items())),
            "unique_score_distribution": distribution([details[key]["best_score"] for key in unique]),
            "unique_rank_distribution": distribution([details[key]["rank"] for key in unique]),
            "unique_instances": [
                {"image_id": key[0], "class_name": key[1], "gt_index": key[2], **details[key]}
                for key in sorted(unique)
            ],
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
