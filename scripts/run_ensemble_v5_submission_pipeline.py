"""Run resumable two-model Test inference and build audited submissions."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_ensemble_submission_from_caches import build_ensemble_submissions
from scripts.run_final_submission_pipeline import package_submission, run_logged
from steel_defect.runtime import sha256_file
from steel_defect.voc import IMAGE_EXTENSIONS


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"
EXPECTED_IMAGES = 788


def validate_cache(path: Path, expected_sha: str, imgsz: int) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("type") != "candidate_cache_header" or header.get("schema") != 3:
            raise ValueError(f"invalid candidate cache header: {path}")
        if header.get("weights_sha256") != expected_sha:
            raise ValueError(f"candidate cache weight mismatch: {path}")
        options = header.get("options", {})
        expected_options = {
            "tile_layout": "semifinal_grid",
            "tile_size": 1024,
            "overlap": 0.25,
            "imgsz": imgsz,
            "batch": 1,
            "conf": 0.00001,
            "local_iou": 0.70,
            "half": True,
            "tta": False,
            "deblur": False,
            "clahe": False,
            "global_pass": True,
            "max_det": 2000,
        }
        mismatched = {
            key: {"expected": value, "actual": options.get(key)}
            for key, value in expected_options.items()
            if options.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"candidate cache inference policy mismatch: {path}: {mismatched}"
            )
        for line in handle:
            json.loads(line)
            count += 1
    return count


def verify_package() -> None:
    manifest_path = ROOT / "ENSEMBLE_V5_PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, digest in manifest.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"package member failed SHA256 verification: {relative}")


def generate_cache(
    project: Path,
    source: Path,
    weight: Path,
    cache: Path,
    expected_sha: str,
    imgsz: int,
    log: Path,
) -> None:
    if cache.is_file():
        if validate_cache(cache, expected_sha, imgsz) != EXPECTED_IMAGES:
            raise ValueError(f"partial candidate cache exists: {cache}")
        print(f"Reuse verified candidate cache: {cache}", flush=True)
        return
    temporary = cache.with_name(cache.stem + ".tmp.jsonl.gz")
    if temporary.exists():
        raise FileExistsError(f"partial temporary cache exists: {temporary}")
    run_logged(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/infer.py"),
            "--weights",
            str(weight),
            "--source",
            str(source),
            "--cache-only",
            "--candidate-cache",
            str(temporary),
            "--tile-layout",
            "semifinal_grid",
            "--imgsz",
            str(imgsz),
            "--batch",
            "1",
            "--conf",
            "0.00001",
            "--local-iou",
            "0.70",
            "--device",
            "0",
            "--half",
            "--global-pass",
            "--max-det",
            "2000",
        ],
        project,
        log,
    )
    if validate_cache(temporary, expected_sha, imgsz) != EXPECTED_IMAGES:
        raise RuntimeError("generated candidate cache does not contain all Test images")
    temporary.replace(cache)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    source = (args.source or project / "data/semifinal/test").resolve()
    weight_a = project / "runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt"
    weight_b = project / "runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt"
    output = project / "runs/semifinal/ensemble_v5_submission"
    profile = ROOT / "configs/inference/ensemble_v5_fusion.json"

    verify_package()
    for weight, digest in ((weight_a, MODEL_A_SHA256), (weight_b, MODEL_B_SHA256)):
        if not weight.is_file() or sha256_file(weight) != digest:
            raise ValueError(f"missing or wrong model weight: {weight}")
    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} Test images, found {len(images)}")
    if len({path.name for path in images}) != len(images):
        raise ValueError("Test image basenames are not unique")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("ensemble inference requires the requested RTX 4090")

    preflight = {
        "gpu": torch.cuda.get_device_name(0),
        "source": str(source),
        "images": len(images),
        "model_a": str(weight_a),
        "model_a_sha256": MODEL_A_SHA256,
        "model_a_imgsz": 1024,
        "model_b": str(weight_b),
        "model_b_sha256": MODEL_B_SHA256,
        "model_b_imgsz": 1280,
        "tta": False,
        "profile": str(profile),
        "profile_sha256": sha256_file(profile),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    cache_a = output / "test_model_a_1e5.jsonl.gz"
    legacy_a = project / "runs/semifinal/final_submission_v1/test_candidates_1e5.jsonl.gz"
    if not cache_a.exists() and legacy_a.is_file():
        if validate_cache(legacy_a, MODEL_A_SHA256, 1024) == EXPECTED_IMAGES:
            cache_a = legacy_a
            print(f"Reuse verified legacy Model A cache: {cache_a}", flush=True)
    generate_cache(project, source, weight_a, cache_a, MODEL_A_SHA256, 1024, output / "model_a.log")

    cache_b = output / "test_model_b_1e5.jsonl.gz"
    generate_cache(project, source, weight_b, cache_b, MODEL_B_SHA256, 1280, output / "model_b.log")

    global_folder = output / "global_fusion"
    routed_folder = output / "class_routed"
    global_submission = global_folder / "submission.json"
    routed_submission = routed_folder / "submission.json"
    report_path = output / "build_report.json"
    if not global_submission.exists() and not routed_submission.exists() and not report_path.exists():
        report = build_ensemble_submissions(
            cache_a,
            cache_b,
            profile,
            global_submission,
            routed_submission,
            report_path,
        )
    else:
        if not global_submission.is_file() or not routed_submission.is_file() or not report_path.is_file():
            raise FileExistsError("partial ensemble build exists; move the output directory aside")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("model_a_sha256") != MODEL_A_SHA256 or report.get("model_b_sha256") != MODEL_B_SHA256:
            raise ValueError("stale ensemble build report")

    summary = {
        "preflight": preflight,
        "cache_a": str(cache_a),
        "cache_a_sha256": sha256_file(cache_a),
        "cache_b": str(cache_b),
        "cache_b_sha256": sha256_file(cache_b),
        "variants": {},
        "submission_order": ["global_fusion", "class_routed"],
        "rule_warning": "Confirm that multi-model inference is permitted before official submission.",
    }
    for name, submission in (
        ("global_fusion", global_submission),
        ("class_routed", routed_submission),
    ):
        folder = submission.parent
        validation = folder / "validation.json"
        run_logged(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/validate_large_submission.py"),
                str(submission),
                "--source",
                str(source),
                "--report",
                str(validation),
                "--exclude-category",
                "qilie",
            ],
            project,
            output / f"validate_{name}.log",
        )
        checked = json.loads(validation.read_text(encoding="utf-8"))
        if checked["expected_images"] != EXPECTED_IMAGES:
            raise RuntimeError(f"submission image-set validation failed: {name}")
        zip_path = output / f"semifinal_ensemble_v5_{name}.zip"
        zip_sha = package_submission(submission, zip_path)
        summary["variants"][name] = {
            "submission": str(submission),
            "submission_sha256": sha256_file(submission),
            "predictions": checked["predictions"],
            "images_with_predictions": checked["images_with_predictions"],
            "images_without_predictions": len(checked["missing_images"]),
            "zip": str(zip_path),
            "zip_sha256": zip_sha,
            "class_counts": report["variants"]["global" if name == "global_fusion" else "routed"]["class_counts"],
        }

    summary_path = output / "ENSEMBLE_V5_SUBMISSION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
