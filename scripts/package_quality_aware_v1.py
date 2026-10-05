"""Build and verify the code-only quality-aware detector experiment package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_quality_aware_v1_fix3_20260928.zip"
PREFIX = "semifinal_quality_aware_v1_fix3_20260928/"
FILES = [
    Path("README_QUALITY_AWARE_V1_4090.md"),
    Path("configs/data/semifinal_full_retrain_v3.yaml"),
    Path("configs/train/semifinal_quality_bce_control_4090.yaml"),
    Path("configs/train/semifinal_quality_varifocal_4090.yaml"),
    Path("scripts/prepare_data.py"),
    Path("scripts/run_full_retrain_pipeline.py"),
    Path("scripts/run_quality_aware_experiment.py"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("scripts/select_full_retrain_checkpoint.py"),
    Path("scripts/train.py"),
    Path("scripts/verify_quality_checkpoint_compatibility.py"),
    Path("tests/test_quality_aware_experiment.py"),
    Path("tests/test_quality_trainer.py"),
    *sorted(path.relative_to(ROOT) for path in (ROOT / "steel_defect").glob("*.py")),
]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite release: {OUTPUT}")
    for relative in FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".pt", ".jpg", ".jpeg", ".png", ".xml"}:
            raise ValueError(f"weights or data cannot enter package: {relative}")
    manifest = {
        relative.as_posix(): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FILES
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, PREFIX + relative.as_posix())
        archive.writestr(
            PREFIX + "QUALITY_AWARE_PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("package CRC verification failed")
        if len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("unexpected package member count")
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise RuntimeError("duplicate package member")
        for relative, digest in manifest.items():
            payload = archive.read(PREFIX + relative)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise RuntimeError(f"package digest mismatch: {relative}")
        forbidden = {".pt", ".jpg", ".jpeg", ".png", ".xml"}
        if any(Path(name).suffix.lower() in forbidden for name in archive.namelist()):
            raise RuntimeError("package contains weights or image/annotation data")
        if any(Path(name).name.lower() == "submission.json" for name in archive.namelist()):
            raise RuntimeError("package contains a leaderboard submission")
    print(json.dumps({
        "archive": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "members": len(FILES) + 1,
        "contains_weights_images_or_submission": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

