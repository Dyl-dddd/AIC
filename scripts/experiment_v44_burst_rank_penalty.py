"""Frozen V12 OOF ablation: penalize class bursts before global AP ranking.

No Test labels are used. This script does not emit a submission.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steel_defect.metrics import _match_class, average_precision
from scripts.audit_v41_v12_bottlenecks import OOF, ground_truth
from scripts.audit_v43_test_matched_dev import cell_id

OUT = ROOT / "runs/semifinal/v44_burst_rank_penalty_20261004"
MATCHED = ROOT / "runs/semifinal/v43_test_matched_dev_20261004/report.json"
CLASS = "mamianmakeng"


def evaluate(predictions: list[dict], gt: dict, ids: set[str], class_name: str) -> dict:
    selected = [p for p in predictions if p["image_id"] in ids]
    truth = {k: v for k, v in gt.items() if k in ids}
    tp, fp, n_gt = _match_class(selected, truth, class_name, .5)
    return {
        "gt": int(n_gt), "predictions": len(selected), "tp": int(tp.sum()),
        "top100_tp": int(tp[:100].sum()),
        "ap50": average_precision(tp, fp, n_gt) if n_gt else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    match = json.loads(MATCHED.read_text(encoding="utf-8"))
    matched_ids = {cell_id(name) for name in match["matched_dev_files"]}
    result = {}
    for view in ("original", "grid_crops"):
        gt = ground_truth(view)
        original = json.loads((OOF / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        numeric_ids = {k for k in gt if not k.startswith("C")}
        strata = {"numeric": numeric_ids, "all": set(gt)}
        if view == "grid_crops":
            strata["test_matched_numeric"] = matched_ids
        class_original = [p for p in original if p["category_name"] == CLASS]
        counts = Counter(p["image_id"] for p in class_original)
        numeric_counts = np.array([counts[k] for k in numeric_ids], dtype=float)
        cutoff = float(np.percentile(numeric_counts, 90))
        result[view] = {"numeric_images": len(numeric_ids), "numeric_burst_p90": cutoff, "trials": {}}
        for alpha in (0, .25, .5, 1.0):
            # Only images above the frozen numeric-dev P90 are affected.
            modified = []
            for p in class_original:
                image_count = counts[p["image_id"]]
                factor = min(1., cutoff / max(image_count, 1)) ** alpha
                modified.append({**p, "score": float(p["score"]) * factor})
            result[view]["trials"][str(alpha)] = {
                key: evaluate(modified, gt, ids, CLASS) for key, ids in strata.items()
            }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
