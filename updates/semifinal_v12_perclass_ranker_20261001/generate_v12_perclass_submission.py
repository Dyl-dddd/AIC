"""Create the V12 independent-per-class ranker submission from V10 caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from xgboost import XGBClassifier


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
        "--v10-root",
        type=Path,
        default=Path("/mnt/proj/iron/semifinal_v10_nonlinear_ranker_fix2_20260930"),
    )
    args = parser.parse_args()
    project = args.project_root.resolve()
    v10 = args.v10_root.resolve()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(v10))

    from scripts.rescore_submission_with_nonlinear_support import (  # noqa: PLC0415
        add_context,
        add_support,
        identity_digest,
        image_sizes,
    )
    from scripts.rescore_submission_with_varifocal import load_predictions  # noqa: PLC0415
    from steel_defect.runtime import sha256_file  # noqa: PLC0415

    spec_path = root / "v12_perclass_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    baseline_path = project / "runs/semifinal/ensemble_v7_tta_submission/primary/submission.json"
    vf_path = project / "runs/semifinal/varifocal_oof_b75_submission/varifocal_test_predictions.json"
    cached = project / "runs/semifinal/v10_nonlinear_ranker_submission"
    p1_path = cached / "v9_p1_test_predictions.json"
    p2_path = cached / "v9_p2_test_predictions.json"
    source = project / "data/semifinal/test"
    validator = v10 / "scripts/validate_large_submission.py"
    required = (baseline_path, vf_path, p1_path, p2_path, validator)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required cached inputs:\n" + "\n".join(missing))

    names = tuple(spec["feature_names"])
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

    probabilities = np.zeros(len(baseline), dtype=np.float64)
    covered = np.zeros(len(baseline), dtype=bool)
    for class_name, details in spec["classes"].items():
        model_path = root / "models" / details["model"]
        if sha256_file(model_path).lower() != details["sha256"].lower():
            raise ValueError(f"model SHA256 mismatch: {class_name}")
        indices = np.asarray(
            [index for index, row in enumerate(baseline) if row["category_name"] == class_name],
            dtype=np.int64,
        )
        if not len(indices):
            raise ValueError(f"no Test predictions for class: {class_name}")
        model = XGBClassifier()
        model.load_model(model_path)
        if int(model.n_features_in_) != len(names):
            raise ValueError(f"feature mismatch: {class_name}")
        probabilities[indices] = model.predict_proba(matrix[indices])[:, 1]
        covered[indices] = True
        print(f"RANK class={class_name} predictions={len(indices)} beta={details['beta']}", flush=True)
    if not bool(np.all(covered)):
        unknown = sorted({baseline[i]["category_name"] for i in np.flatnonzero(~covered)})
        raise ValueError(f"uncovered categories: {unknown}")

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

    output_root = project / "runs/semifinal/v12_perclass_ranker_submission"
    output_root.mkdir(parents=True, exist_ok=True)
    submission = output_root / "submission.json"
    temporary = submission.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(submission)
    validation = output_root / "validation.json"
    subprocess.run(
        [
            sys.executable, "-u", str(validator), str(submission),
            "--source", str(source), "--report", str(validation),
            "--exclude-category", "qilie",
        ],
        cwd=project,
        check=True,
    )
    archive = output_root / "semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip"
    package_submission(submission, archive)
    report = {
        "method": "independent per-class XGBoost rankers",
        "class_betas": {name: row["beta"] for name, row in spec["classes"].items()},
        "predictions": len(output),
        "membership_and_geometry_preserved": True,
        "identity_sha256": identity,
        "submission_sha256": sha256_file(submission),
        "archive_sha256": sha256_file(archive),
        "strict_group_oof": spec["strict_group_oof"],
        "archive": str(archive),
    }
    (output_root / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
