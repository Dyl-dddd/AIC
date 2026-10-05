"""Build one-class V33 score-reranking submission from frozen V12 baseline.

All boxes, image IDs, classes, prediction count, and non-mamian scores remain
bit-for-bit unchanged. Train labels only fit image context models; no Test labels.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_v32_image_context_rerank import image_features

SPLIT = ROOT / "data/official_v2_20260831b/splits.json"
CROP_CACHE = ROOT / "runs/semifinal/v33_crop_context_20261003/train_features.npz"
BASELINE = Path(r"D:/新建文件夹 (8)/semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip")
TEST_ROOT = ROOT / "data/semifinal/test"
OUTPUT = Path(r"D:/新建文件夹 (8)/semifinal_v33_mamian_context_rerank_SUBMIT_ONLY.zip")
REPORT = ROOT / "runs/semifinal/v33_crop_context_20261003/submission_build.json"
CLASS = "mamianmakeng"


def fit_full(split: dict) -> XGBClassifier:
    names = [n for n in split["splits"]["train"] if not n.startswith("C")]
    records = split["records"]
    x = np.stack([image_features(np.frombuffer(base64.b64decode(records[n]["thumbnail"]),
                                               np.uint8).reshape(64, 64)) for n in names])
    y = np.asarray([any(a[0] == CLASS for a in records[n]["annotations"]) for n in names],
                   dtype=np.int32)
    pos = int(y.sum())
    model = XGBClassifier(n_estimators=180, max_depth=2, learning_rate=0.035,
                          min_child_weight=12, subsample=0.85, colsample_bytree=0.8,
                          reg_lambda=12, objective="binary:logistic", eval_metric="logloss",
                          n_jobs=4, random_state=20261003,
                          scale_pos_weight=min(8.0, (len(y) - pos) / pos))
    model.fit(x, y)
    return model


def fit_crop() -> XGBClassifier:
    cache = np.load(CROP_CACHE)
    x, y = cache["x"], cache["y"]
    pos = int(y.sum())
    model = XGBClassifier(n_estimators=180, max_depth=2, learning_rate=0.035,
                          min_child_weight=12, subsample=0.85, colsample_bytree=0.8,
                          reg_lambda=12, objective="binary:logistic", eval_metric="logloss",
                          n_jobs=4, random_state=20261003,
                          scale_pos_weight=min(8.0, (len(y) - pos) / pos))
    model.fit(x, y)
    return model


def main() -> None:
    if not BASELINE.exists():
        raise FileNotFoundError(BASELINE)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    full_model, crop_model = fit_full(split), fit_crop()
    with zipfile.ZipFile(BASELINE) as zf:
        if zf.namelist() != ["submission.json"]:
            raise RuntimeError(f"Unexpected baseline ZIP members: {zf.namelist()}")
        baseline_bytes = zf.read("submission.json")
    baseline = json.loads(baseline_bytes)
    needed = {r["image_id"] for r in baseline if r["category_name"] == CLASS}
    probs: dict[str, float] = {}
    weights: dict[str, float] = {}
    type_counts = {"full": 0, "crop": 0}
    for index, image_id in enumerate(sorted(needed), 1):
        path = TEST_ROOT / image_id
        if not path.is_file():
            raise FileNotFoundError(path)
        gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Unable to decode {path}")
        shape = gray.shape
        if shape == (3000, 4096):
            model, weight, kind = full_model, 0.25, "full"
        elif shape == (1516, 1387):
            model, weight, kind = crop_model, 0.1, "crop"
        else:
            raise RuntimeError(f"Unexpected Test dimensions for {image_id}: {shape}")
        thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        probs[image_id] = float(model.predict_proba(image_features(thumb)[None, :])[0, 1])
        weights[image_id] = weight
        type_counts[kind] += 1
        if index % 100 == 0:
            print("Test images", index, "/", len(needed), flush=True)
    output = []
    modified = 0
    for row in baseline:
        if row["category_name"] == CLASS:
            image_id = row["image_id"]
            score = float(row["score"]) * max(1e-4, probs[image_id]) ** weights[image_id]
            if not 0 < score <= 1:
                raise RuntimeError((image_id, score))
            output.append({**row, "score": score})
            modified += 1
        else:
            output.append(row)
    if len(output) != len(baseline):
        raise RuntimeError("Prediction count changed")
    for before, after in zip(baseline, output, strict=True):
        if before["image_id"] != after["image_id"] or before["category_name"] != after["category_name"] \
                or before["bbox"] != after["bbox"]:
            raise RuntimeError("A prediction identity or box changed")
        if before["category_name"] != CLASS and before["score"] != after["score"]:
            raise RuntimeError("A non-target class score changed")
    payload = json.dumps(output, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("submission.json", payload)
    with zipfile.ZipFile(OUTPUT) as zf:
        if zf.testzip() is not None or zf.namelist() != ["submission.json"]:
            raise RuntimeError("Invalid output ZIP")
        verify = json.loads(zf.read("submission.json"))
        if len(verify) != len(baseline):
            raise RuntimeError("ZIP roundtrip row count changed")
    report = {
        "baseline": str(BASELINE), "baseline_zip_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "output": str(OUTPUT), "output_zip_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "predictions": len(output), "rescored_mamian_predictions": modified,
        "target_images": len(needed), "target_image_types": type_counts,
        "score_rule": "full score * p_mamian^0.25; crop score * p_mamian^0.1",
        "non_target_scores_and_all_geometry_identical": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
