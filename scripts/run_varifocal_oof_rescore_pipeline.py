"""Build the frozen Varifocal-OOF score-transfer submission on RTX 4090."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_submission_with_varifocal import build as build_rescored
from scripts.run_ensemble_v6_consensus_pipeline import package_submission, sha256_file
from scripts.run_final_submission_pipeline import run_logged
from steel_defect.voc import IMAGE_EXTENSIONS


VARIFOCAL_SHA256 = "4228f473d7618275e6c7efea6ad28118a6da2d2a2c9d8422298fd0078430a4f6"
EXPECTED_IMAGES = 788


def count_images(source: Path) -> int:
    images = [
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(images) != EXPECTED_IMAGES or len({path.name for path in images}) != EXPECTED_IMAGES:
        raise ValueError("Test source must contain 788 uniquely named images")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    source = (args.source or project / "data/semifinal/test").resolve()
    weight_options = (
        project / "runs/semifinal/quality_aware_v1_fix3_20260928_pipeline/final/varifocal.pt",
        project / "runs/semifinal/quality_aware_v1_fix3_20260928_varifocal/weights/last.pt",
    )
    weight = next(
        (path for path in weight_options if path.is_file() and sha256_file(path) == VARIFOCAL_SHA256),
        weight_options[0],
    )
    profile = ROOT / "configs/inference/varifocal_oof_b75.json"
    output = project / "runs/semifinal/varifocal_oof_b75_submission"
    if not weight.is_file() or sha256_file(weight) != VARIFOCAL_SHA256:
        raise ValueError(f"missing or wrong Varifocal checkpoint: {weight}")
    if not profile.is_file():
        raise FileNotFoundError(profile)
    images = count_images(source)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Varifocal Test inference requires RTX 4090")

    preflight = {
        "gpu": torch.cuda.get_device_name(0),
        "source": str(source),
        "images": images,
        "training": False,
        "v7_membership_and_geometry_frozen": True,
        "varifocal_weight": str(weight),
        "varifocal_sha256": VARIFOCAL_SHA256,
        "profile_sha256": sha256_file(profile),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    v7_submission = project / "runs/semifinal/ensemble_v7_tta_submission/primary/submission.json"
    if not v7_submission.is_file():
        run_logged(
            [
                sys.executable, "-u", str(ROOT / "scripts/run_ensemble_v7_tta_submission_pipeline.py"),
                "--project-root", str(project), "--source", str(source),
            ],
            project,
            output / "v7_pipeline.log",
        )
    if not v7_submission.is_file():
        raise FileNotFoundError("V7 primary submission was not produced")

    output.mkdir(parents=True, exist_ok=True)
    quality = output / "varifocal_test_predictions.json"
    quality_audit = output / "varifocal_test_predictions.audit.json"
    if quality.is_file() or quality_audit.is_file():
        if not (quality.is_file() and quality_audit.is_file()):
            raise FileExistsError("partial Varifocal inference output exists")
        audit = json.loads(quality_audit.read_text(encoding="utf-8"))
        expected = {
            "weight_sha256": VARIFOCAL_SHA256,
            "source": str(source),
            "output_sha256": sha256_file(quality),
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError("stale Varifocal inference output")
    else:
        temporary = output / "varifocal_test_predictions.tmp.json"
        if temporary.exists():
            raise FileExistsError(f"preserve and move partial file: {temporary}")
        run_logged(
            [
                sys.executable, "-u", str(ROOT / "scripts/infer.py"),
                "--weights", str(weight), "--source", str(source),
                "--output", str(temporary), "--tile-layout", "semifinal_grid",
                "--imgsz", "1024", "--batch", "1", "--conf", "0.0001",
                "--iou", "0.55", "--local-iou", "0.70", "--device", "0",
                "--half", "--edge-penalty", "0.85", "--global-pass",
                "--max-det", "2000",
            ],
            project,
            output / "varifocal_inference.log",
        )
        temporary.replace(quality)
        quality_audit.write_text(json.dumps({
            "weight_sha256": VARIFOCAL_SHA256,
            "source": str(source),
            "output": str(quality),
            "output_sha256": sha256_file(quality),
            "options": {
                "tile_layout": "semifinal_grid", "imgsz": 1024, "batch": 1,
                "conf": 0.0001, "iou": 0.55, "local_iou": 0.70,
                "half": True, "edge_penalty": 0.85, "global_pass": True,
                "tta": False, "max_det": 2000,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    submission = output / "submission.json"
    build_report = output / "build_report.json"
    if not submission.exists() and not build_report.exists():
        built = build_rescored(v7_submission, quality, profile, submission, build_report)
    elif submission.is_file() and build_report.is_file():
        built = json.loads(build_report.read_text(encoding="utf-8"))
        expected = {
            "baseline_sha256": sha256_file(v7_submission),
            "quality_sha256": sha256_file(quality),
            "profile_sha256": sha256_file(profile),
            "output_sha256": sha256_file(submission),
        }
        if any(built.get(key) != value for key, value in expected.items()):
            raise ValueError("stale rescored submission")
    else:
        raise FileExistsError("partial rescored submission exists")

    validation = output / "validation.json"
    subprocess.run([
        sys.executable, "-u", str(ROOT / "scripts/validate_large_submission.py"),
        str(submission), "--source", str(source), "--report", str(validation),
        "--exclude-category", "qilie",
    ], cwd=project, check=True)
    checked = json.loads(validation.read_text(encoding="utf-8"))
    if checked["expected_images"] != EXPECTED_IMAGES:
        raise RuntimeError("submission image-set validation failed")
    zip_path = output / "semifinal_varifocal_oof_b75.zip"
    summary = {
        "preflight": preflight,
        "development_oof_weighted_delta": 0.1447926796950867,
        "development_oof_original_delta": 0.13003466893594862,
        "development_oof_grid_delta": 0.16879904359666997,
        "build": built,
        "validation": checked,
        "zip": str(zip_path),
        "zip_sha256": package_submission(submission, zip_path),
        "official_score": None,
        "warning": "OOF proxy improvement is not an official leaderboard guarantee.",
    }
    summary_path = output / "VARIFOCAL_OOF_B75_SUBMISSION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
