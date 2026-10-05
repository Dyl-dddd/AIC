"""Run V10 VF+V9 nonlinear score ranking and package one submission JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_submission_with_nonlinear_support import build as build_rescored
from steel_defect.runtime import sha256_file
from steel_defect.voc import IMAGE_EXTENSIONS


EXPECTED_IMAGES = 788
P1_SHA256 = "9a4ad2b8fe5489e1e5aa89a4ba377afc57ec85f8e29aa19b0efb307b4f20ef68"
P2_SHA256 = "224153cc092f88ffed4043c401a2ce0d545726e47b45301dab66fc85f2f3f846"
SUBMISSION_BLENDS = (0.65, 0.80, 0.75, 0.50, 0.40)
OOF_DELTAS = {
    0.65: {"weighted": 0.355860, "original": 0.377257, "grid_crops": 0.321054},
    0.80: {"weighted": 0.378819, "original": 0.428655, "grid_crops": 0.297751},
    0.75: {"weighted": 0.361968, "original": 0.396819, "grid_crops": 0.305277},
    0.50: {"weighted": 0.304539, "original": 0.304521, "grid_crops": 0.304568},
    0.40: {"weighted": 0.245877, "original": 0.246956, "grid_crops": 0.244122},
}


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def package_submission(source: Path, destination: Path) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(source, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("invalid submit-only ZIP")
    temporary.replace(destination)
    return sha256_file(destination)


def count_images(source: Path) -> int:
    images = [
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(images) != EXPECTED_IMAGES or len({path.name for path in images}) != EXPECTED_IMAGES:
        raise ValueError("Test source must contain 788 uniquely named images")
    return len(images)


def infer_quality(
    project: Path, source: Path, weight: Path, digest: str,
    output: Path, audit_path: Path, log: Path,
) -> None:
    if output.is_file() or audit_path.is_file():
        if not (output.is_file() and audit_path.is_file()):
            raise FileExistsError(f"partial quality inference output: {output}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected = {
            "weight_sha256": digest,
            "source": str(source),
            "output_sha256": sha256_file(output),
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError(f"stale quality inference output: {output}")
        print(f"Reuse verified quality output: {output}", flush=True)
        return
    temporary = output.with_name(output.stem + ".tmp.json")
    if temporary.exists():
        raise FileExistsError(f"preserve and move partial file: {temporary}")
    run_logged(
        [
            sys.executable, "-u", str(ROOT / "scripts/infer.py"),
            "--weights", str(weight), "--source", str(source),
            "--output", str(temporary), "--tile-layout", "semifinal_grid",
            "--imgsz", "1280", "--batch", "1", "--conf", "0.0001",
            "--iou", "0.55", "--local-iou", "0.70", "--device", "0",
            "--half", "--edge-penalty", "0.85", "--global-pass",
            "--max-det", "2000",
        ],
        project,
        log,
    )
    temporary.replace(output)
    audit_path.write_text(json.dumps({
        "weight": str(weight),
        "weight_sha256": digest,
        "source": str(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "options": {
            "tile_layout": "semifinal_grid", "imgsz": 1280, "batch": 1,
            "conf": 0.0001, "iou": 0.55, "local_iou": 0.70,
            "half": True, "edge_penalty": 0.85, "global_pass": True,
            "tta": False, "max_det": 2000,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = (args.source or project / "data/semifinal/test").resolve()
    output = project / "runs/semifinal/v10_nonlinear_ranker_submission"
    config = ROOT / "configs/inference/v10_nonlinear_ranker.json"
    model = ROOT / "models/v10_nonlinear_ranker.ubj"
    baseline = project / "runs/semifinal/ensemble_v7_tta_submission/primary/submission.json"
    vf = project / "runs/semifinal/varifocal_oof_b75_submission/varifocal_test_predictions.json"
    p1_weight = project / "runs/semifinal/v9_model_b_quality_p1/weights/last.pt"
    p2_weight = project / "runs/semifinal/v9_model_b_quality_p2/weights/last.pt"

    if not config.is_file() or not model.is_file():
        raise FileNotFoundError("V10 config/model missing from package")
    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if sha256_file(model) != config_payload["deployment"]["model_sha256"]:
        raise ValueError("V10 deployment model SHA256 mismatch")
    for weight, digest in ((p1_weight, P1_SHA256), (p2_weight, P2_SHA256)):
        if not weight.is_file() or sha256_file(weight) != digest:
            raise ValueError(f"missing or wrong V9 checkpoint: {weight}")
    images = count_images(source)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V10 P1/P2 Test inference requires RTX 4090")
    preflight = {
        "gpu": torch.cuda.get_device_name(0),
        "source": str(source),
        "images": images,
        "training": False,
        "membership_and_geometry_frozen": True,
        "p1_sha256": P1_SHA256,
        "p2_sha256": P2_SHA256,
        "model_sha256": sha256_file(model),
        "config_sha256": sha256_file(config),
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return
    output.mkdir(parents=True, exist_ok=True)

    if not baseline.is_file():
        raise FileNotFoundError(f"run V7 pipeline first; missing {baseline}")
    if not vf.is_file():
        raise FileNotFoundError(
            "missing Varifocal Test evidence; rerun the existing Varifocal OOF B75 "
            f"pipeline first: {vf}"
        )

    p1 = output / "v9_p1_test_predictions.json"
    p2 = output / "v9_p2_test_predictions.json"
    infer_quality(project, source, p1_weight, P1_SHA256, p1, output / "v9_p1_test_predictions.audit.json", output / "v9_p1_inference.log")
    infer_quality(project, source, p2_weight, P2_SHA256, p2, output / "v9_p2_test_predictions.audit.json", output / "v9_p2_inference.log")

    variants = {}
    for blend in SUBMISSION_BLENDS:
        tag = f"b{int(round(blend * 100)):03d}"
        folder = output / "variants" / tag
        folder.mkdir(parents=True, exist_ok=True)
        submission = folder / "submission.json"
        report = folder / "build_report.json"
        if not submission.exists() and not report.exists():
            built = build_rescored(
                baseline, vf, p1, p2, source, model, config,
                submission, report, blend,
            )
        elif submission.is_file() and report.is_file():
            built = json.loads(report.read_text(encoding="utf-8"))
            expected = {
                "blend": blend,
                "baseline_sha256": sha256_file(baseline),
                "vf_sha256": sha256_file(vf),
                "p1_sha256": sha256_file(p1),
                "p2_sha256": sha256_file(p2),
                "output_sha256": sha256_file(submission),
            }
            if any(built.get(key) != value for key, value in expected.items()):
                raise ValueError(f"stale V10 submission output: {tag}")
        else:
            raise FileExistsError(f"partial V10 build output exists: {tag}")

        validation = folder / "validation.json"
        subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts/validate_large_submission.py"),
            str(submission), "--source", str(source), "--report", str(validation),
            "--exclude-category", "qilie",
        ], cwd=project, check=True)
        checked = json.loads(validation.read_text(encoding="utf-8"))
        if checked["expected_images"] != EXPECTED_IMAGES:
            raise RuntimeError(f"submission image-set validation failed: {tag}")
        zip_path = output / f"semifinal_v10_{tag}_SUBMIT_ONLY.zip"
        variants[tag] = {
            "blend": blend,
            "oof_delta_vs_v7": OOF_DELTAS[blend],
            "build": built,
            "validation": checked,
            "zip": str(zip_path),
            "zip_sha256": package_submission(submission, zip_path),
            "official_score": None,
        }

    summary = {
        "preflight": preflight,
        "development_oof": config_payload["development_oof"],
        "submission_order": [f"b{int(round(value * 100)):03d}" for value in SUBMISSION_BLENDS],
        "variants": variants,
        "warning": "OOF proxy improvement is not an official leaderboard guarantee.",
    }
    summary_path = output / "V10_NONLINEAR_RANKER_SUBMISSION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
