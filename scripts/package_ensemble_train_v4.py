"""Build and verify the semifinal YOLO11m ensemble-training package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_ensemble_train_v4_20260926.zip"
PREFIX = "semifinal_ensemble_train_v4_20260926/"
YOLO11M = Path("yolo11m.pt")
YOLO11M_SHA = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
FILES = [
    Path("README_ENSEMBLE_TRAIN_V4_4090.md"),
    YOLO11M,
    Path("configs/data/semifinal_full_retrain_v3.yaml"),
    Path("configs/train/semifinal_ensemble_v4_yolo11m_phase1_4090.yaml"),
    Path("configs/train/semifinal_ensemble_v4_yolo11m_phase2_4090.yaml"),
    Path("scripts/prepare_data.py"),
    Path("scripts/train.py"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("scripts/select_full_retrain_checkpoint.py"),
    Path("scripts/run_full_retrain_pipeline.py"),
    Path("scripts/run_ensemble_v4_training_pipeline.py"),
    Path("tests/test_full_retrain_v3.py"),
    Path("tests/test_ensemble_v4.py"),
    *sorted(path.relative_to(ROOT) for path in (ROOT / "steel_defect").glob("*.py")),
]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite release: {OUTPUT}")
    if len(FILES) != len(set(FILES)):
        raise ValueError("duplicate package member")

    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".xml"}:
            raise ValueError(f"training data cannot enter package: {relative}")
        if source.suffix.lower() == ".pt" and relative != YOLO11M:
            raise ValueError(f"unexpected checkpoint in package: {relative}")

    if hashlib.sha256((ROOT / YOLO11M).read_bytes()).hexdigest() != YOLO11M_SHA:
        raise ValueError("bundled YOLO11m SHA256 mismatch")

    manifest = {
        relative.as_posix(): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FILES
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, PREFIX + relative.as_posix())
        archive.writestr(
            PREFIX + "ENSEMBLE_TRAIN_PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("archive integrity check failed")
        for relative, digest in manifest.items():
            actual = hashlib.sha256(archive.read(PREFIX + relative)).hexdigest()
            if actual != digest:
                raise RuntimeError(f"archive digest mismatch: {relative}")

    print(
        json.dumps(
            {
                "archive": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                "files": len(FILES) + 1,
                "bundled_pretrained": YOLO11M.as_posix(),
                "bundled_pretrained_sha256": YOLO11M_SHA,
                "contains_training_data": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
