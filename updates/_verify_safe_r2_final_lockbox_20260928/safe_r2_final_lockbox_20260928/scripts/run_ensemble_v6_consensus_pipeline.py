"""Reuse existing Test caches and build the V6 source-aware consensus submission."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_source_aware_submission_from_caches import build_submission
from steel_defect.classes import CLASS_NAMES
from steel_defect.voc import IMAGE_EXTENSIONS


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"
EXPECTED_IMAGES = 788


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package() -> None:
    manifest_path = ROOT / "ENSEMBLE_V6_PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"package member failed SHA256 verification: {relative}")


def validate_cache(path: Path, expected_sha: str, imgsz: int) -> int:
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
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if header.get("type") != "candidate_cache_header" or header.get("schema") != 3:
            raise ValueError(f"invalid candidate cache: {path}")
        if header.get("weights_sha256") != expected_sha or header.get("classes") != CLASS_NAMES:
            raise ValueError(f"candidate cache model/classes mismatch: {path}")
        options = header.get("options", {})
        mismatched = {key: (value, options.get(key)) for key, value in expected_options.items()
                      if options.get(key) != value}
        if mismatched:
            raise ValueError(f"candidate cache policy mismatch: {path}: {mismatched}")
        for line in handle:
            json.loads(line)
            count += 1
    return count


def package_submission(source: Path, destination: Path) -> str:
    temporary = destination.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(source, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("invalid submission ZIP")
    temporary.replace(destination)
    return sha256_file(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = (args.source or project / "data/semifinal/test").resolve()
    output = project / "runs/semifinal/ensemble_v6_consensus_submission"
    profile = ROOT / "configs/inference/ensemble_v6_consensus.json"
    cache_a_options = (
        project / "runs/semifinal/ensemble_v5_submission/test_model_a_1e5.jsonl.gz",
        project / "runs/semifinal/final_submission_v1/test_candidates_1e5.jsonl.gz",
    )
    cache_a = next((path for path in cache_a_options if path.is_file()), None)
    cache_b = project / "runs/semifinal/ensemble_v5_submission/test_model_b_1e5.jsonl.gz"

    verify_package()
    if cache_a is None or not cache_b.is_file():
        raise FileNotFoundError("V5 Model A/B Test candidate caches are required")
    if validate_cache(cache_a, MODEL_A_SHA256, 1024) != EXPECTED_IMAGES:
        raise ValueError("Model A cache is incomplete")
    if validate_cache(cache_b, MODEL_B_SHA256, 1280) != EXPECTED_IMAGES:
        raise ValueError("Model B cache is incomplete")
    images = sorted(path for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(images) != EXPECTED_IMAGES or len({path.name for path in images}) != EXPECTED_IMAGES:
        raise ValueError("Test source must contain 788 uniquely named images")
    preflight = {
        "source": str(source), "images": len(images),
        "cache_a": str(cache_a), "cache_a_sha256": sha256_file(cache_a),
        "cache_b": str(cache_b), "cache_b_sha256": sha256_file(cache_b),
        "profile_sha256": sha256_file(profile), "gpu_required": False,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    submission = output / "submission.json"
    build_report = output / "build_report.json"
    if not submission.exists() and not build_report.exists():
        built = build_submission(cache_a, cache_b, profile, submission, build_report)
    elif submission.is_file() and build_report.is_file():
        built = json.loads(build_report.read_text(encoding="utf-8"))
        if built.get("model_a_sha256") != MODEL_A_SHA256 or built.get("model_b_sha256") != MODEL_B_SHA256:
            raise ValueError("stale V6 build output")
    else:
        raise FileExistsError("partial V6 build exists; preserve and move it aside")

    validation = output / "validation.json"
    subprocess.run([
        sys.executable, "-u", str(ROOT / "scripts/validate_large_submission.py"),
        str(submission), "--source", str(source), "--report", str(validation),
        "--exclude-category", "qilie",
    ], cwd=project, check=True)
    checked = json.loads(validation.read_text(encoding="utf-8"))
    if checked["expected_images"] != EXPECTED_IMAGES:
        raise RuntimeError("submission image-set validation failed")
    zip_path = output / "semifinal_ensemble_v6_consensus.zip"
    summary = {
        "preflight": preflight,
        "build": built,
        "validation": checked,
        "zip": str(zip_path),
        "zip_sha256": package_submission(submission, zip_path),
        "protected_official_score": 66.96,
    }
    summary_path = output / "ENSEMBLE_V6_CONSENSUS_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

