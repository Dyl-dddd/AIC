"""Build and verify the code-only V9 quality-localizer experiment package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_v9_quality_localizer_20260930.zip"
PREFIX = "semifinal_v9_quality_localizer_20260930/"
FILES = [
    Path("README_V9_QUALITY_LOCALIZER_4090.md"),
    Path("configs/train/semifinal_v9_model_b_quality_p1_4090.yaml"),
    Path("configs/train/semifinal_v9_model_b_quality_p2_4090.yaml"),
    Path("scripts/run_full_retrain_pipeline.py"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("scripts/run_v9_quality_localizer_experiment.py"),
    Path("scripts/select_full_retrain_checkpoint.py"),
    Path("scripts/train.py"),
    Path("tests/test_quality_trainer.py"),
    Path("tests/test_v9_quality_localizer.py"),
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
            PREFIX + "V9_QUALITY_LOCALIZER_PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("package integrity check failed")
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise RuntimeError("duplicate package member")
        for relative, expected in manifest.items():
            if hashlib.sha256(archive.read(PREFIX + relative)).hexdigest() != expected:
                raise RuntimeError(f"package digest mismatch: {relative}")
        if any(Path(name).suffix.lower() in {".pt", ".jpg", ".jpeg", ".png", ".xml"}
               for name in archive.namelist()):
            raise RuntimeError("weights or data entered package")
        if any(Path(name).name.lower() == "submission.json" for name in archive.namelist()):
            raise RuntimeError("submission entered experiment package")
    print(json.dumps({
        "archive": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "members": len(FILES) + 1,
        "contains_weights_images_or_submission": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
