"""Run one final-weight Test inference and build three audited semifinal submissions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_final_submission_from_cache import build_submission
from steel_defect.runtime import sha256_file
from steel_defect.voc import IMAGE_EXTENSIONS


WEIGHT_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
EXPECTED_IMAGES = 788


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def count_cache(path: Path) -> int:
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
        if header.get("type") != "candidate_cache_header" or header.get("schema") != 3:
            raise ValueError("invalid candidate cache header")
        if header.get("weights_sha256") != WEIGHT_SHA256:
            raise ValueError("candidate cache was generated with a different weight")
        for line in handle:
            json.loads(line)
            count += 1
    return count


def package_submission(source: Path, destination: Path) -> str:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(source, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"invalid submission ZIP: {temporary}")
    temporary.replace(destination)
    return sha256_file(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = (args.source or project / "data/semifinal/Test").resolve()
    weight = project / "runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt"
    output = project / "runs/semifinal/final_submission_v1"
    python = sys.executable

    manifest_path = ROOT / "FINAL_INFERENCE_PACKAGE_MANIFEST.json"
    if manifest_path.is_file():
        for relative, digest in json.loads(manifest_path.read_text(encoding="utf-8")).items():
            path = ROOT / relative
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"package member failed SHA256 verification: {relative}")
    if not weight.is_file() or sha256_file(weight) != WEIGHT_SHA256:
        raise ValueError(f"missing or wrong final weight: {weight}")
    images = sorted(path for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"expected {EXPECTED_IMAGES} Test images, found {len(images)} under {source}")
    if len({path.name for path in images}) != len(images):
        raise ValueError("Test image basenames are not unique")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("final inference requires the requested RTX 4090")
    preflight = {"gpu": torch.cuda.get_device_name(0), "weight": str(weight),
        "weight_sha256": WEIGHT_SHA256, "source": str(source), "images": len(images),
        "generation_conf": 0.00001, "local_nms_iou": 0.70,
        "final_nms_iou": 0.68, "edge_penalty": 0.5}
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    cache = output / "test_candidates_1e5.jsonl.gz"
    if cache.is_file():
        if count_cache(cache) != EXPECTED_IMAGES:
            raise ValueError(f"partial candidate cache exists; move it aside: {cache}")
        print(f"Reuse verified candidate cache: {cache}", flush=True)
    else:
        temporary = output / "test_candidates_1e5.tmp.jsonl.gz"
        run_logged([python, "-u", str(ROOT / "scripts/infer.py"),
            "--weights", str(weight), "--source", str(source), "--cache-only",
            "--candidate-cache", str(temporary), "--tile-layout", "semifinal_grid",
            "--imgsz", "1024", "--batch", "1", "--conf", "0.00001",
            "--local-iou", "0.70", "--device", "0", "--half", "--global-pass",
            "--max-det", "2000"], project, output / "inference.log")
        if count_cache(temporary) != EXPECTED_IMAGES:
            raise RuntimeError("generated candidate cache does not contain all Test images")
        temporary.replace(cache)

    variants = {
        "recall_safe": {"floor": 0.0001, "thresholds": None},
        "dev_calibrated": {"floor": 0.00001,
            "thresholds": ROOT / "configs/inference/full_retrain_v3_final_thresholds.json"},
        "ultra_recall_1e5": {"floor": 0.00001, "thresholds": None},
    }
    summary = {"preflight": preflight, "candidate_cache": str(cache),
               "candidate_cache_sha256": sha256_file(cache), "variants": {}}
    for name, profile in variants.items():
        folder = output / name
        submission = folder / "submission.json"
        report = folder / "build_report.json"
        validation = folder / "validation.json"
        if not submission.is_file() or not report.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            temp_submission = folder / "submission.tmp.json"
            temp_report = folder / "build_report.tmp.json"
            built = build_submission(cache, temp_submission, temp_report,
                profile["thresholds"], profile["floor"], 0.68, 0.5, WEIGHT_SHA256)
            temp_submission.replace(submission)
            temp_report.replace(report)
        else:
            built = json.loads(report.read_text(encoding="utf-8"))
            if built.get("weights_sha256") != WEIGHT_SHA256:
                raise ValueError(f"stale variant report: {report}")
        run_logged([python, "-u", str(ROOT / "scripts/validate_large_submission.py"),
            str(submission), "--source", str(source), "--report", str(validation),
            "--exclude-category", "qilie"], project, output / f"validate_{name}.log")
        checked = json.loads(validation.read_text(encoding="utf-8"))
        # The competition format is a flat array of detections. Images with no
        # retained detections are represented by the absence of rows; inserting
        # a dummy row would create a false positive. Therefore missing_images is
        # diagnostic information, not a submission-format failure.
        if checked["expected_images"] != EXPECTED_IMAGES:
            raise RuntimeError(f"submission image-set validation failed: {name}")
        zip_path = output / f"semifinal_{name}.zip"
        zip_sha = package_submission(submission, zip_path)
        summary["variants"][name] = {"submission": str(submission),
            "submission_sha256": sha256_file(submission), "predictions": checked["predictions"],
            "images_with_predictions": checked["images_with_predictions"],
            "images_without_predictions": len(checked["missing_images"]),
            "zip": str(zip_path), "zip_sha256": zip_sha, "thresholds": built["thresholds"]}
    summary_path = output / "FINAL_SUBMISSION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
