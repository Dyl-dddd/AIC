"""Train the accepted texture-only OOF module and rescore V12 yanghuatiepi."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from scripts.analyze_final_submission_postprocess import load_cell
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_v19_texture_reranker import CELLS, IMAGES, ROOT, fit_texture, texture_features
from scripts.analyze_v9_support_oof import initial_records
from scripts.analyze_varifocal_oof_transfer import load_groups


TARGET = "yanghuatiepi"
BLEND = 0.15
DEV_SPLITS = ROOT / "data/official_v2_20260831b/splits.json"
TEST_IMAGES = ROOT / "data/semifinal/test"
SOURCE_ZIP = Path(r"D:\新建文件夹 (8)\semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip")
OUTPUT = ROOT / "runs/semifinal/v19_texture_submission_20261002"
DELIVERY = Path(r"D:\新建文件夹 (8)\semifinal_v19_texture_yanghuatiepi_v2_SUBMIT_ONLY.zip")


def identity(items: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(json.dumps(
            (item["image_id"], item["category_name"], item["bbox"]),
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))
    return digest.hexdigest()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    oof = json.loads((ROOT / "runs/semifinal/v19_texture_reranker_20261002/results.json").read_text())
    for view in ("original", "grid_crops"):
        delta = oof["classes"][TARGET][view]["blends"][str(BLEND)]["delta"]
        if delta < 0.012:
            raise RuntimeError(f"OOF guard failed: {view} {delta:+.6f}")
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("invalid V12 source ZIP")
        baseline = json.loads(archive.read("submission.json"))
    source_identity = identity(baseline)
    groups = load_groups(DEV_SPLITS)
    train_rows = []
    for view in ("original", "grid_crops"):
        a = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        b = load_cell(CELLS / f"model_b_tta_s1280_{view}")
        predictions = [row for row in fuse_v7(a, b) if row["category_name"] == TARGET]
        train_rows.extend(initial_records(predictions, a["ground_truth"], groups))
    x_train = texture_features(train_rows, IMAGES)
    y_train = np.asarray([row["label"] for row in train_rows], dtype=np.int32)
    model = fit_texture(x_train, y_train, 6401)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / "texture_yanghuatiepi.ubj"
    model.save_model(model_path)
    print(f"TRAIN rows={len(train_rows)} positives={int(y_train.sum())}", flush=True)
    target_indices = [index for index, item in enumerate(baseline) if item["category_name"] == TARGET]
    test_rows = [{"prediction": baseline[index]} for index in target_indices]
    x_test = texture_features(test_rows, TEST_IMAGES)
    probabilities = model.predict_proba(x_test)[:, 1]
    output = [dict(item) for item in baseline]
    for index, probability in zip(target_indices, probabilities, strict=True):
        score = max(float(output[index]["score"]), 1e-12)
        output[index]["score"] = float(score ** (1.0 - BLEND) * max(float(probability), 1e-12) ** BLEND)
    if identity(output) != source_identity:
        raise RuntimeError("candidate membership/geometry changed")
    if any(output[index]["score"] != baseline[index]["score"] for index, item in enumerate(output) if item["category_name"] != TARGET):
        raise RuntimeError("non-target class changed")
    submission = OUTPUT / "submission.json"
    submission.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    validation = OUTPUT / "validation.json"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/validate_large_submission.py"), str(submission),
        "--source", str(TEST_IMAGES), "--report", str(validation), "--exclude-category", "qilie",
    ], check=True)
    if not json.loads(validation.read_text())["valid"]:
        raise RuntimeError("submission validation failed")
    temporary = DELIVERY.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(submission, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("invalid output ZIP")
    temporary.replace(DELIVERY)
    report = {
        "method": "group-OOF-accepted image texture residual for yanghuatiepi",
        "blend": BLEND,
        "test_labels_used": False,
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": sha(SOURCE_ZIP),
        "identity_sha256": source_identity,
        "test_predictions": len(output),
        "target_predictions": len(target_indices),
        "model_sha256": sha(model_path),
        "submission_sha256": sha(submission),
        "archive": str(DELIVERY),
        "archive_sha256": sha(DELIVERY),
        "oof_delta_ap50": {view: oof["classes"][TARGET][view]["blends"][str(BLEND)]["delta"] for view in ("original", "grid_crops")},
        "official_score_known": False,
    }
    (OUTPUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
