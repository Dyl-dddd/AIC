"""Build class-adaptive V11 submissions from cached V10 test predictions.

This script does not run GPU inference.  It reconstructs the frozen V10
nonlinear probability once, then applies auditable per-class blend strengths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from xgboost import XGBClassifier


POLICIES = {
    # Strict group-OOF weighted best per class.  This is the primary release.
    "classwise_primary": {
        "jieba": 0.90,
        "zonglie": 0.70,
        "jiaza": 0.80,
        "yiwuyaru": 0.90,
        "huashang": 0.50,
        "mamianmakeng": 0.85,
        "yanghuatiepi": 0.70,
        "gunyin": 0.80,
    },
    # Use only if primary beats the current 67.77 by at least 0.08.
    "classwise_right_shift": {
        "jieba": 0.95,
        "zonglie": 0.75,
        "jiaza": 0.85,
        "yiwuyaru": 0.95,
        "huashang": 0.55,
        "mamianmakeng": 0.90,
        "yanghuatiepi": 0.75,
        "gunyin": 0.85,
    },
    # Use if primary improves by less than 0.08 or regresses.
    "classwise_shrink085": {
        "jieba": 0.875,
        "zonglie": 0.775,
        "jiaza": 0.825,
        "yiwuyaru": 0.875,
        "huashang": 0.675,
        "mamianmakeng": 0.85,
        "yanghuatiepi": 0.775,
        "gunyin": 0.825,
    },
}


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_submission(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(source, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"invalid submit-only archive: {temporary}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument(
        "--v10-root",
        type=Path,
        default=Path("/mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930"),
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    v10 = args.v10_root.resolve()
    sys.path.insert(0, str(v10))

    from scripts.rescore_submission_with_nonlinear_support import (  # noqa: PLC0415
        add_context,
        add_support,
        identity_digest,
        image_sizes,
    )
    from scripts.rescore_submission_with_varifocal import load_predictions  # noqa: PLC0415
    from steel_defect.runtime import sha256_file  # noqa: PLC0415

    baseline_path = project / "runs/semifinal/ensemble_v7_tta_submission/primary/submission.json"
    vf_path = project / "runs/semifinal/varifocal_oof_b75_submission/varifocal_test_predictions.json"
    cached = project / "runs/semifinal/v10_nonlinear_ranker_submission"
    p1_path = cached / "v9_p1_test_predictions.json"
    p2_path = cached / "v9_p2_test_predictions.json"
    source = project / "data/semifinal/test"
    model_path = v10 / "models/v10_nonlinear_ranker.ubj"
    config_path = v10 / "configs/inference/v10_nonlinear_ranker.json"
    validator = v10 / "scripts/validate_large_submission.py"
    required = (baseline_path, vf_path, p1_path, p2_path, model_path, config_path, validator)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required cached inputs:\n" + "\n".join(missing))

    config = json.loads(config_path.read_text(encoding="utf-8"))
    deployment = config["deployment"]
    if sha256_file(model_path) != deployment["model_sha256"]:
        raise ValueError("V10 model SHA256 mismatch")
    names = tuple(deployment["feature_names"])
    baseline = load_predictions(baseline_path)
    features = []
    for row in baseline:
        base = max(float(row["score"]), 1e-12)
        feature = {"log_base_score": float(np.log(base))}
        for prefix in ("vf", "p1", "p2"):
            feature.update({
                f"{prefix}_present": 0.0,
                f"{prefix}_iou": 0.0,
                f"log_{prefix}_score": float(np.log(1e-12)),
                f"{prefix}_log_advantage": float(np.log(1e-12) - np.log(base)),
            })
        features.append(feature)
    for path, prefix in ((vf_path, "vf"), (p1_path, "p1"), (p2_path, "p2")):
        add_support(features, baseline, load_predictions(path), prefix)
    sizes = image_sizes(source, {str(row["image_id"]) for row in baseline})
    add_context(features, baseline, sizes)
    matrix = np.asarray([[row[name] for name in names] for row in features], dtype=np.float32)
    model = XGBClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(matrix)[:, 1]

    output_root = project / "runs/semifinal/v11_classwise_ranker_submission"
    output_root.mkdir(parents=True, exist_ok=True)
    identity = identity_digest(baseline)
    summary = {"method": "class-adaptive geometric score blend", "variants": {}}
    for tag, class_blends in POLICIES.items():
        rows = []
        for row, probability in zip(baseline, probabilities, strict=True):
            item = dict(row)
            blend = float(class_blends.get(str(item["category_name"]), 0.85))
            base = max(float(item["score"]), 1e-12)
            calibrated = max(float(probability), 1e-12)
            item["score"] = base ** (1.0 - blend) * calibrated ** blend
            rows.append(item)
        if identity_digest(rows) != identity:
            raise RuntimeError(f"membership or geometry changed for {tag}")
        folder = output_root / tag
        folder.mkdir(parents=True, exist_ok=True)
        submission = folder / "submission.json"
        report = folder / "build_report.json"
        atomic_json(submission, rows)
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(validator),
                str(submission),
                "--source",
                str(source),
                "--report",
                str(folder / "validation.json"),
                "--exclude-category",
                "qilie",
            ],
            cwd=project,
            check=True,
        )
        archive = output_root / f"semifinal_v11_{tag}_SUBMIT_ONLY.zip"
        package_submission(submission, archive)
        report_payload = {
            "policy": tag,
            "class_blends": class_blends,
            "predictions": len(rows),
            "membership_and_geometry_preserved": True,
            "identity_sha256": identity,
            "baseline_sha256": sha256_file(baseline_path),
            "vf_sha256": sha256_file(vf_path),
            "p1_sha256": sha256_file(p1_path),
            "p2_sha256": sha256_file(p2_path),
            "model_sha256": sha256_file(model_path),
            "submission_sha256": sha256_file(submission),
            "archive_sha256": sha256_file(archive),
        }
        report.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["variants"][tag] = {"archive": str(archive), **report_payload}
        print(f"READY {tag}: {archive}", flush=True)
    summary_path = output_root / "V11_CLASSWISE_SUBMISSION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
