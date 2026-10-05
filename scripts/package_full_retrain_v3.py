"""Build and verify the isolated full-retraining upload archive."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_full_retrain_v3_fix1_20260925.zip"
PREFIX = "semifinal_full_retrain_v3_fix1_20260925/"
FILES = [
    Path("README_FULL_RETRAIN_V3_FIX1_4090.md"),
    Path("configs/data/semifinal_full_retrain_v3.yaml"),
    Path("configs/train/semifinal_full_retrain_phase1_4090.yaml"),
    Path("configs/train/semifinal_full_retrain_phase2_4090.yaml"),
    Path("scripts/prepare_data.py"),
    Path("scripts/train.py"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("scripts/select_full_retrain_checkpoint.py"),
    Path("scripts/run_full_retrain_pipeline.py"),
    Path("tests/test_full_retrain_v3.py"),
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
        if source.suffix.lower() in {".pt", ".jpg", ".jpeg", ".png", ".xml"}:
            raise ValueError(f"weights/data cannot enter code-only package: {relative}")
    manifest = {relative.as_posix(): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                for relative in FILES}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, PREFIX + relative.as_posix())
        archive.writestr(PREFIX + "FULL_RETRAIN_PACKAGE_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC failed")
        if len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("archive member-count mismatch")
        for relative, digest in manifest.items():
            if hashlib.sha256(archive.read(PREFIX + relative)).hexdigest() != digest:
                raise RuntimeError(f"archive digest mismatch: {relative}")
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), "files": len(FILES) + 1,
        "contains_weights_or_images": False, "full_training_entrypoint": True},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

