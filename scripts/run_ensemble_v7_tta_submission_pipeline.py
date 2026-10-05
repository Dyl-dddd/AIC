"""Run resumable dual-model native TTA on Test and build two audited V7 submissions."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_source_aware_submission_from_caches import build_submission
from scripts.run_ensemble_v6_consensus_pipeline import package_submission, sha256_file
from scripts.run_final_submission_pipeline import run_logged
from steel_defect.classes import CLASS_NAMES
from steel_defect.voc import IMAGE_EXTENSIONS


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"
EXPECTED_IMAGES = 788


def verify_package() -> None:
    manifest_path = ROOT / "ENSEMBLE_V7_TTA_PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"package member failed SHA256 verification: {relative}")


def validate_tta_cache(path: Path, expected_sha: str, imgsz: int) -> int:
    expected_options = {
        "tile_layout": "semifinal_grid",
        "tile_size": 1024,
        "overlap": 0.25,
        "imgsz": imgsz,
        "batch": 1,
        "conf": 0.00001,
        "local_iou": 0.70,
        "half": True,
        "tta": True,
        "deblur": False,
        "clahe": False,
        "global_pass": True,
        "max_det": 2000,
    }
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("type") != "candidate_cache_header" or header.get("schema") != 3:
            raise ValueError(f"invalid candidate cache: {path}")
        if header.get("weights_sha256") != expected_sha or header.get("classes") != CLASS_NAMES:
            raise ValueError(f"candidate cache model/classes mismatch: {path}")
        options = header.get("options", {})
        mismatched = {
            key: {"expected": value, "actual": options.get(key)}
            for key, value in expected_options.items()
            if options.get(key) != value
        }
        if mismatched:
            raise ValueError(f"candidate cache policy mismatch: {path}: {mismatched}")
        for line in handle:
            json.loads(line)
            count += 1
    return count


def generate_tta_cache(
    project: Path,
    source: Path,
    weight: Path,
    cache: Path,
    expected_sha: str,
    imgsz: int,
    log: Path,
) -> None:
    if cache.is_file():
        if validate_tta_cache(cache, expected_sha, imgsz) != EXPECTED_IMAGES:
            raise ValueError(f"partial candidate cache exists: {cache}")
        print(f"Reuse verified TTA candidate cache: {cache}", flush=True)
        return
    temporary = cache.with_name(cache.stem + ".tmp.jsonl.gz")
    if temporary.exists():
        raise FileExistsError(f"partial temporary cache exists: {temporary}")
    run_logged(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/infer.py"),
            "--weights", str(weight),
            "--source", str(source),
            "--cache-only",
            "--candidate-cache", str(temporary),
            "--tile-layout", "semifinal_grid",
            "--imgsz", str(imgsz),
            "--batch", "1",
            "--conf", "0.00001",
            "--local-iou", "0.70",
            "--device", "0",
            "--half",
            "--global-pass",
            "--tta",
            "--max-det", "2000",
        ],
        project,
        log,
    )
    if validate_tta_cache(temporary, expected_sha, imgsz) != EXPECTED_IMAGES:
        raise RuntimeError(f"generated TTA cache is incomplete: {temporary}")
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
    output = project / "runs/semifinal/ensemble_v7_tta_submission"

    verify_package()
    for weight, digest in ((weight_a, MODEL_A_SHA256), (weight_b, MODEL_B_SHA256)):
        if not weight.is_file() or sha256_file(weight) != digest:
            raise ValueError(f"missing or wrong protected weight: {weight}")
    images = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) != EXPECTED_IMAGES or len({path.name for path in images}) != EXPECTED_IMAGES:
        raise ValueError("Test source must contain 788 uniquely named images")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("dual-model TTA Test inference requires RTX 4090")

    profiles = {
        "primary": ROOT / "configs/inference/ensemble_v7_tta_primary.json",
        "conservative": ROOT / "configs/inference/ensemble_v7_tta_conservative.json",
    }
    preflight = {
        "gpu": torch.cuda.get_device_name(0),
        "source": str(source),
        "images": len(images),
        "training": False,
        "tta": True,
        "model_a": {"path": str(weight_a), "sha256": MODEL_A_SHA256, "imgsz": 1024},
        "model_b": {"path": str(weight_b), "sha256": MODEL_B_SHA256, "imgsz": 1280},
        "profiles": {name: sha256_file(path) for name, path in profiles.items()},
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    cache_a = output / "test_model_a_tta_1e5.jsonl.gz"
    legacy_a_options = (
        project / "runs/semifinal/final_submission_tta_v1/test_candidates_tta_1e5.jsonl.gz",
        project / "runs/semifinal/ensemble_v7_tta_submission/test_model_a_tta_1e5.jsonl.gz",
    )
    reusable_a = next(
        (
            path for path in legacy_a_options
            if path.is_file() and validate_tta_cache(path, MODEL_A_SHA256, 1024) == EXPECTED_IMAGES
        ),
        None,
    )
    if reusable_a is not None:
        cache_a = reusable_a
        print(f"Reuse verified Model A TTA cache: {cache_a}", flush=True)
    else:
        generate_tta_cache(
            project, source, weight_a, cache_a, MODEL_A_SHA256, 1024,
            output / "model_a_tta.log",
        )

    cache_b = output / "test_model_b_tta_1e5.jsonl.gz"
    generate_tta_cache(
        project, source, weight_b, cache_b, MODEL_B_SHA256, 1280,
        output / "model_b_tta.log",
    )

    summary = {
        "preflight": preflight,
        "cache_a": str(cache_a),
        "cache_a_sha256": sha256_file(cache_a),
        "cache_b": str(cache_b),
        "cache_b_sha256": sha256_file(cache_b),
        "variants": {},
        "submission_order": ["primary", "conservative"],
        "protected_official_score": 66.96,
    }
    for name, profile in profiles.items():
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        submission = folder / "submission.json"
        build_report = folder / "build_report.json"
        if not submission.exists() and not build_report.exists():
            built = build_submission(cache_a, cache_b, profile, submission, build_report)
        elif submission.is_file() and build_report.is_file():
            built = json.loads(build_report.read_text(encoding="utf-8"))
            if built.get("profile_sha256") != sha256_file(profile):
                raise ValueError(f"stale V7 {name} build output")
        else:
            raise FileExistsError(f"partial V7 {name} build exists; preserve and move it aside")

        validation = folder / "validation.json"
        subprocess.run(
            [
                sys.executable, "-u", str(ROOT / "scripts/validate_large_submission.py"),
                str(submission), "--source", str(source), "--report", str(validation),
                "--exclude-category", "qilie",
            ],
            cwd=project,
            check=True,
        )
        checked = json.loads(validation.read_text(encoding="utf-8"))
        if checked["expected_images"] != EXPECTED_IMAGES:
            raise RuntimeError(f"submission image-set validation failed: {name}")
        zip_path = output / f"semifinal_ensemble_v7_tta_{name}.zip"
        summary["variants"][name] = {
            "build": built,
            "validation": checked,
            "zip": str(zip_path),
            "zip_sha256": package_submission(submission, zip_path),
        }

    summary_path = output / "ENSEMBLE_V7_TTA_SUBMISSION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
