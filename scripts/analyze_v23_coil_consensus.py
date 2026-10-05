"""Cross-frame, same-camera defect support on frozen V12 OOF candidates.

Unsupervised: neighboring images provide only prediction scores, never labels.
This is a hypothesis screen; development gains are not official Test gains.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

from scripts.analyze_final_submission_postprocess import load_cell, metrics50


ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OUT = ROOT / "runs/semifinal/v23_coil_consensus_20261003"
BETAS = (0.5, 1.0, 2.0, 4.0)
TOP_K = 20
MIN_SUPPORT_SCORE = 0.10
X_SCALE = 90.0


def source_name(image_id: str) -> str:
    return image_id.split("::", 1)[0]


def sequence_key(source: str) -> tuple[str, str] | None:
    first = re.match(r"^(C\d+)_V(\d+)_F\d+", source, flags=re.IGNORECASE)
    if first:
        return first.group(1).upper(), first.group(2)
    second = re.match(r"^(\d+)[-_]Raw(\d+)[-_]f_\d+", source, flags=re.IGNORECASE)
    if second:
        return second.group(1), second.group(2)
    return None


def full_center_x(row: dict) -> float:
    center = 0.5 * (row["bbox"][0] + row["bbox"][2])
    suffix = row["image_id"].split("::", 1)
    if len(suffix) == 2:
        center += int(suffix[1].split("_", 1)[0][1:])
    return center


def support_scores(rows: list[dict]) -> np.ndarray:
    groups: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        source = source_name(row["image_id"])
        sequence = sequence_key(source)
        if sequence is None:
            continue
        groups[(*sequence, row["category_name"])][source].append(index)
    centers = np.asarray([full_center_x(row) for row in rows], dtype=np.float32)
    scores = np.asarray([row["score"] for row in rows], dtype=np.float32)
    support = np.zeros(len(rows), dtype=np.float32)
    used = 0
    for sources in groups.values():
        if len(sources) < 2:
            continue
        used += sum(map(len, sources.values()))
        shortlist = {}
        for source, indices in sources.items():
            ranked = sorted(indices, key=lambda index: scores[index], reverse=True)
            shortlist[source] = np.asarray(
                [index for index in ranked if scores[index] >= MIN_SUPPORT_SCORE][:TOP_K],
                dtype=np.int32,
            )
        for source, indices in sources.items():
            if not indices:
                continue
            local = np.asarray(indices, dtype=np.int32)
            best_matches = []
            for other, selected in shortlist.items():
                if other == source or not len(selected):
                    continue
                distance = (centers[local, None] - centers[selected][None, :]) / X_SCALE
                affinity = np.exp(-0.5 * np.square(distance)) * scores[selected][None, :]
                best_matches.append(affinity.max(axis=1))
            if best_matches:
                stacked = np.stack(best_matches, axis=1)
                stacked.sort(axis=1)
                support[local] = stacked[:, -min(2, stacked.shape[1]):].sum(axis=1) / 2.0
    print(f"SEQUENCE_SUPPORT rows={len(rows)} eligible={used} "
          f"nonzero={int(np.count_nonzero(support))}", flush=True)
    return support


def rerank(rows: list[dict], support: np.ndarray, beta: float,
           enabled: set[str] | None = None) -> list[dict]:
    output = []
    for row, extra in zip(rows, support, strict=True):
        item = dict(row)
        if enabled is None or row["category_name"] in enabled:
            base = min(max(float(item["score"]), 1e-8), 1 - 1e-8)
            logit = math.log(base / (1.0 - base)) + beta * float(extra)
            item["score"] = 1.0 / (1.0 + math.exp(-logit))
        output.append(item)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"protocol": "same-coil, same-camera unlabeled prediction consensus",
              "parameters": {"top_k": TOP_K, "min_support_score": MIN_SUPPORT_SCORE,
                             "x_scale": X_SCALE}, "views": {}}
    for view in ("original", "grid_crops"):
        print(f"LOAD {view}", flush=True)
        rows = json.loads((OOF / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        cell = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        support = support_scores(rows)
        np.save(OUT / f"{view}_support.npy", support)
        baseline = metrics50(rows, cell["ground_truth"], cell["sizes"])
        variants = {}
        for beta in BETAS:
            metric = metrics50(rerank(rows, support, beta), cell["ground_truth"], cell["sizes"])
            variants[str(beta)] = {
                "score": metric["score"], "score_delta": metric["score"] - baseline["score"],
                "map50_delta": metric["map50"] - baseline["map50"],
                "tp": metric["tp"], "fp": metric["fp"], "fn": metric["fn"],
                "class_ap_delta": {name: metric["per_class"][name]["ap50"]
                                   - baseline["per_class"][name]["ap50"]
                                   for name in baseline["per_class"]},
            }
            print(f"RESULT {view} beta={beta} delta={variants[str(beta)]['score_delta']:+.6f}", flush=True)
        result["views"][view] = {"baseline": baseline, "variants": variants,
                                 "eligible_predictions": int(np.count_nonzero(support))}
    (OUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
