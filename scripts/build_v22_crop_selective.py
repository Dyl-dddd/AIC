"""Audit and optionally package a conservative two-class crop rerank.

This is an exploratory submission candidate, not a guaranteed leaderboard gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.evaluate_v20_crop_verifier import CELLS, OOF, load_model, score_candidates
from scripts.v20_crop_verifier import OUT


ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "data/semifinal/test"
V12_SUBMISSION = Path(r"D:\新建文件夹 (8)\semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip")
POLICY = {"huashang": 0.10, "jiaza": 0.10}


def rerank(rows: list[dict], probability: np.ndarray) -> list[dict]:
    if len(rows) != len(probability):
        raise RuntimeError("length mismatch")
    output = []
    for row, extra in zip(rows, probability, strict=True):
        item = dict(row)
        alpha = POLICY.get(row["category_name"], 0.0)
        if alpha:
            item["score"] = float(max(float(row["score"]), 1e-12) ** (1.0 - alpha)
                                  * max(float(extra), 1e-12) ** alpha)
        output.append(item)
    return output


def audit() -> dict:
    result = {"model_sha256": hashlib.sha256((OUT / "best.pt").read_bytes()).hexdigest(),
              "policy": POLICY, "views": {}}
    for view in ("original", "grid_crops"):
        cell = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        rows = json.loads((OOF / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        probability = np.load(OUT / f"{view}_class_probability.npy")
        before = metrics50(rows, cell["ground_truth"], cell["sizes"])
        after = metrics50(rerank(rows, probability), cell["ground_truth"], cell["sizes"])
        result["views"][view] = {"before": before, "after": after,
                                 "score_delta": after["score"] - before["score"],
                                 "map50_delta": after["map50"] - before["map50"]}
        print(f"AUDIT {view}: score_delta={after['score']-before['score']:+.6f} "
              f"mAP_delta={after['map50']-before['map50']:+.6f} "
              f"TP={before['tp']}->{after['tp']}", flush=True)
    (OUT / "selective_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def package() -> None:
    report = audit()
    if not all(row["score_delta"] > 0 and row["after"]["tp"] == row["before"]["tp"]
               for row in report["views"].values()):
        raise RuntimeError("selective policy failed development safety gate")
    with zipfile.ZipFile(V12_SUBMISSION) as archive:
        if archive.namelist() != ["submission.json"]:
            raise RuntimeError("V12 submission archive has unexpected members")
        rows = json.loads(archive.read("submission.json"))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probability = score_candidates(rows, TEST_IMAGES, load_model(), device)
    updated = rerank(rows, probability)
    output = OUT / "semifinal_v22_selective_crop_SUBMIT_ONLY.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", json.dumps(updated, ensure_ascii=False, separators=(",", ":")))
    report["submission"] = {"archive": str(output), "predictions": len(updated),
                            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                            "changed": sum(a["score"] != b["score"] for a, b in zip(rows, updated, strict=True))}
    (OUT / "selective_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["submission"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "package"))
    args = parser.parse_args()
    if args.mode == "audit":
        audit()
    else:
        package()
