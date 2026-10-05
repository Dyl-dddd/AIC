"""Exploratory, frozen-dev-only q-box extent audit on the protected V12 rows.

This uses existing Model A q proposals; it does not alter V12 entries, fit on
Test, or make a submission. The numeric q ground truth is concentrated in one
image, so any apparent gain is too fragile to deploy without another check.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_v37_qilie_integration import score

SOURCE = ROOT / "runs/semifinal/v37_qilie_specialist/integration/best_score_model_q_candidates.json"
V12 = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_original_predictions.json"
SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
OUT = ROOT / "runs/semifinal/v40_qilie_box_extent_20261004/results.json"


def expand(row: dict, factor: float, width: int) -> dict:
    x1, y1, x2, y2 = map(float, row["bbox"])
    center = (x1 + x2) / 2
    half = (x2 - x1) * factor / 2
    changed = dict(row)
    changed["bbox"] = [int(round(max(0, center - half))), int(round(y1)),
                       int(round(min(width, center + half))), int(round(y2))]
    return changed


def main() -> None:
    cache = json.loads(SOURCE.read_text(encoding="utf-8"))
    frozen = json.loads(SPLIT.read_text(encoding="utf-8"))
    names = sorted(frozen["splits"]["dev"])
    assert cache["image_count"] == len(names)
    v12 = json.loads(V12.read_text(encoding="utf-8"))
    assert all(row["category_name"] != "qilie" for row in v12)
    assert len(cache["predictions"]) < 10000
    ground_truth = {}
    for name in names:
        classes = {}
        for class_name, *box in frozen["records"][name]["annotations"]:
            classes.setdefault(class_name, []).append(list(map(float, box)))
        ground_truth[name] = classes
    baseline = score(v12, ground_truth)
    records = {name: frozen["records"][name] for name in names}
    variants = {}
    # Fixed, small morphology-motivated values; no exhaustive tuning on dev.
    for factor in (1.0, 2.0, 3.0, 4.0):
        q_rows = []
        for row in cache["predictions"]:
            name = row["image_id"]
            # All provided Test images use the numeric acquisition family.
            selected = factor if name[0].isdigit() else 1.0
            q_rows.append(expand(row, selected, records[name]["width"]))
        whole = score(v12 + q_rows, ground_truth)
        q = whole["per_class"]["qilie"]
        variants[str(factor)] = {
            "score_proxy": whole["score_proxy"],
            "score_proxy_delta": whole["score_proxy"] - baseline["score_proxy"],
            "tp_delta": whole["tp"] - baseline["tp"],
            "qilie": q,
            "q_candidates": len(q_rows),
        }
        print("FACTOR", factor, variants[str(factor)], flush=True)
    report = {
        "protocol": "Frozen full nine-class dev; unchanged V12 original-view OOF plus Model A q candidates; numeric-only width expansion",
        "baseline": baseline,
        "variants": variants,
        "warning": "Numeric-source dev q targets are eight boxes in one source image; exploratory only, not an official-score estimate or submission.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
