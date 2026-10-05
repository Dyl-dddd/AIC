"""Build the V14 hybrid per-class ranker using V13 localization evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import torch
import ultralytics
from xgboost import XGBClassifier


EXPECTED_IMAGES = 788
EXPECTED_ULTRALYTICS = "8.3.169"
LOC_SHA256 = "f5a0cf206b4ae5970c2cd1334c9279a9d7a47cd789c0f1437bd0db566d6f852e"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(root: Path) -> str:
    path = root / "V14_PACKAGE_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        member = root / relative
        if not member.is_file() or sha256_file(member) != expected:
            raise ValueError(f"package member mismatch: {relative}")
    return sha256_file(path)


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


def count_images(source: Path) -> int:
    images = [path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if len(images) != EXPECTED_IMAGES or len({path.name for path in images}) != EXPECTED_IMAGES:
        raise ValueError("Test source must contain 788 uniquely named images")
    return len(images)


def infer_localizer(
    project: Path, v10: Path, source: Path, weight: Path,
    output: Path, audit_path: Path, log: Path,
) -> None:
    options = {
        "tile_layout": "semifinal_grid", "imgsz": 1536, "batch": 1,
        "conf": 0.0001, "iou": 0.55, "local_iou": 0.70,
        "half": True, "edge_penalty": 0.85, "global_pass": True,
        "tta": False, "max_det": 2000,
    }
    if output.is_file() or audit_path.is_file():
        if not (output.is_file() and audit_path.is_file()):
            raise FileExistsError(f"partial V13 inference output: {output}")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected = {
            "weight_sha256": LOC_SHA256,
            "source": str(source),
            "output_sha256": sha256_file(output),
            "options": options,
        }
        if any(audit.get(key) != value for key, value in expected.items()):
            raise ValueError(f"stale V13 inference output: {output}")
        print(f"Reuse verified V13 localization output: {output}", flush=True)
        return
    temporary = output.with_name(output.stem + ".tmp.json")
    if temporary.exists():
        raise FileExistsError(f"preserve and move partial file before retry: {temporary}")
    run_logged(
        [
            sys.executable, "-u", str(v10 / "scripts/infer.py"),
            "--weights", str(weight), "--source", str(source),
            "--output", str(temporary), "--tile-layout", "semifinal_grid",
            "--imgsz", "1536", "--batch", "1", "--conf", "0.0001",
            "--iou", "0.55", "--local-iou", "0.70", "--device", "0",
            "--half", "--edge-penalty", "0.85", "--global-pass", "--max-det", "2000",
        ],
        project,
        log,
    )
    temporary.replace(output)
    audit_path.write_text(json.dumps({
        "weight": str(weight), "weight_sha256": LOC_SHA256,
        "source": str(source), "output": str(output),
        "output_sha256": sha256_file(output), "options": options,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def package_submission(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(source, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("invalid submit-only ZIP")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument(
        "--v10-root", type=Path,
        default=Path("/mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    v10 = args.v10_root.resolve()
    root = Path(__file__).resolve().parent
    manifest_sha = verify_package(root)
    sys.path.insert(0, str(v10))

    from scripts.rescore_submission_with_nonlinear_support import (  # noqa: PLC0415
        add_context, add_support, identity_digest, image_sizes,
    )
    from scripts.rescore_submission_with_varifocal import load_predictions  # noqa: PLC0415

    spec = json.loads((root / "v14_hybrid_spec.json").read_text(encoding="utf-8"))
    source = project / "data/semifinal/test"
    baseline_path = project / "runs/semifinal/ensemble_v7_tta_submission/primary/submission.json"
    vf_path = project / "runs/semifinal/varifocal_oof_b75_submission/varifocal_test_predictions.json"
    v10_cached = project / "runs/semifinal/v10_nonlinear_ranker_submission"
    p1_path = v10_cached / "v9_p1_test_predictions.json"
    p2_path = v10_cached / "v9_p2_test_predictions.json"
    loc_weight = project / "runs/semifinal/v13_hires_localize_vfl/weights/last.pt"
    validator = v10 / "scripts/validate_large_submission.py"
    infer_script = v10 / "scripts/infer.py"
    required = (baseline_path, vf_path, p1_path, p2_path, loc_weight, validator, infer_script)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input:\n" + "\n".join(missing))
    if sha256_file(loc_weight) != LOC_SHA256:
        raise ValueError("V13 localization checkpoint SHA256 mismatch")
    if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
        raise RuntimeError(f"expected ultralytics {EXPECTED_ULTRALYTICS}, got {ultralytics.__version__}")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V14 V13 inference requires RTX 4090")
    free, total = torch.cuda.mem_get_info()
    if free < 18 * 1024**3:
        raise RuntimeError(f"less than 18 GiB free VRAM: {free / 1024**3:.2f}")
    preflight = {
        "gpu": torch.cuda.get_device_name(0),
        "free_vram_gib": free / 1024**3,
        "total_vram_gib": total / 1024**3,
        "ultralytics": ultralytics.__version__,
        "images": count_images(source),
        "v13_localize_sha256": LOC_SHA256,
        "package_manifest_sha256": manifest_sha,
        "training": False,
        "membership_and_geometry_frozen": True,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    output_root = project / "runs/semifinal/v14_v13_hybrid_ranker_submission"
    output_root.mkdir(parents=True, exist_ok=True)
    loc_path = output_root / "v13_localize_test_predictions.json"
    infer_localizer(
        project, v10, source, loc_weight, loc_path,
        output_root / "v13_localize_test_predictions.audit.json",
        output_root / "v13_localize_inference.log",
    )

    baseline = load_predictions(baseline_path)
    features = []
    for row in baseline:
        base = max(float(row["score"]), 1e-12)
        feature = {"log_base_score": float(np.log(base))}
        for prefix in ("vf", "p1", "p2", "loc"):
            feature.update({
                f"{prefix}_present": 0.0,
                f"{prefix}_iou": 0.0,
                f"log_{prefix}_score": float(np.log(1e-12)),
                f"{prefix}_log_advantage": float(np.log(1e-12) - np.log(base)),
            })
        features.append(feature)
    for path, prefix in ((vf_path, "vf"), (p1_path, "p1"), (p2_path, "p2"), (loc_path, "loc")):
        add_support(features, baseline, load_predictions(path), prefix)
        print(f"MATCH Test support={prefix}", flush=True)
    sizes = image_sizes(source, {str(row["image_id"]) for row in baseline})
    add_context(features, baseline, sizes)

    probabilities = np.zeros(len(baseline), dtype=np.float64)
    covered = np.zeros(len(baseline), dtype=bool)
    for class_name, details in spec["classes"].items():
        names = tuple(spec["feature_sets"][details["feature_set"]])
        indices = np.asarray(
            [index for index, row in enumerate(baseline) if row["category_name"] == class_name],
            dtype=np.int64,
        )
        if not len(indices):
            raise ValueError(f"no Test predictions for class: {class_name}")
        matrix = np.asarray([[features[index][name] for name in names] for index in indices], dtype=np.float32)
        model_path = root / "models" / details["model"]
        if sha256_file(model_path) != details["sha256"]:
            raise ValueError(f"model SHA256 mismatch: {class_name}")
        model = XGBClassifier()
        model.load_model(model_path)
        if int(model.n_features_in_) != len(names):
            raise ValueError(f"feature count mismatch: {class_name}")
        probabilities[indices] = model.predict_proba(matrix)[:, 1]
        covered[indices] = True
        print(
            f"RANK class={class_name} source={details['feature_set']} "
            f"predictions={len(indices)} beta={details['beta']}",
            flush=True,
        )
    if not bool(np.all(covered)):
        raise ValueError("one or more Test categories are not covered")

    output = []
    for row, probability in zip(baseline, probabilities, strict=True):
        item = dict(row)
        beta = float(spec["classes"][str(item["category_name"])]["beta"])
        base = max(float(item["score"]), 1e-12)
        calibrated = max(float(probability), 1e-12)
        item["score"] = base ** (1.0 - beta) * calibrated ** beta
        output.append(item)
    identity = identity_digest(baseline)
    if identity_digest(output) != identity:
        raise RuntimeError("membership or geometry changed")

    submission = output_root / "submission.json"
    temporary = submission.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(submission)
    validation = output_root / "validation.json"
    subprocess.run(
        [
            sys.executable, "-u", str(validator), str(submission),
            "--source", str(source), "--report", str(validation),
            "--exclude-category", "qilie",
        ], cwd=project, check=True,
    )
    archive = output_root / "semifinal_v14_v13_hybrid_ranker_SUBMIT_ONLY.zip"
    package_submission(submission, archive)
    report = {
        "method": "V12/V14 per-class hybrid with V13 localization support",
        "predictions": len(output),
        "membership_and_geometry_preserved": True,
        "identity_sha256": identity,
        "submission_sha256": sha256_file(submission),
        "archive_sha256": sha256_file(archive),
        "strict_group_oof": spec["strict_group_oof"],
        "class_sources": {name: row["feature_set"] for name, row in spec["classes"].items()},
        "archive": str(archive),
    }
    (output_root / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
