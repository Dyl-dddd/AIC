"""Evaluate frozen V12/V22 candidates separately by acquisition source.

Competition Test images all use the numeric filename family; the official
development set also contains a distinct C-prefixed family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.build_v22_crop_selective import rerank as v22_rerank
from scripts.analyze_v23_coil_consensus import rerank as v23_rerank


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
V20 = ROOT / "runs/semifinal/v20_crop_verifier_20261002"
V23 = ROOT / "runs/semifinal/v23_coil_consensus_20261003"
OUT = ROOT / "runs/semifinal/v24_domain_strata_20261003"


def source(image_id: str) -> str:
    return image_id.split("::", 1)[0]


def family(image_id: str) -> str:
    return "C" if source(image_id).startswith("C") else "numeric"


def subset_metrics(rows: list[dict], cell: dict, selected: str) -> dict:
    ground_truth = {key: value for key, value in cell["ground_truth"].items()
                    if family(key) == selected}
    sizes = {key: value for key, value in cell["sizes"].items() if family(key) == selected}
    subset = [row for row in rows if family(row["image_id"]) == selected]
    return metrics50(subset, ground_truth, sizes)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"test_family": "numeric_only", "views": {}}
    for view in ("original", "grid_crops"):
        cell = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        rows = json.loads((OOF / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        probability = np.load(V20 / f"{view}_class_probability.npy")
        support = np.load(V23 / f"{view}_support.npy")
        candidates = {
            "V12_baseline": rows,
            "V22_selective_crop": v22_rerank(rows, probability),
            "V23_jiaza_coil_b1": v23_rerank(rows, support, 1.0, {"jiaza"}),
        }
        report["views"][view] = {}
        for selected in ("numeric", "C"):
            metrics = {name: subset_metrics(data, cell, selected) for name, data in candidates.items()}
            base = metrics["V12_baseline"]
            report["views"][view][selected] = {
                name: {"score": metric["score"], "score_delta": metric["score"]-base["score"],
                       "map50": metric["map50"], "map50_delta": metric["map50"]-base["map50"],
                       "tp": metric["tp"], "fp": metric["fp"], "fn": metric["fn"],
                       "per_class": metric["per_class"]}
                for name, metric in metrics.items()
            }
            print(f"{view} {selected}: " + ", ".join(
                f"{name}={metric['score']:.4f}({metric['score']-base['score']:+.4f})"
                for name, metric in metrics.items()), flush=True)
    (OUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
