"""Create and verify the isolated stage-2 GPU evaluation archive."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_stage2_eval_20260924.zip"
PREFIX = "semifinal_stage2_20260924/"
FILES = [
    Path("README_STAGE2_4090.md"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("scripts/run_semifinal_stage2.py"),
    Path("tests/test_stage1_geometry.py"),
    *sorted(p.relative_to(ROOT) for p in (ROOT / "steel_defect").glob("*.py")),
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite release: {OUTPUT}")
    for relative in FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix.lower() in {".pt", ".jpg", ".jpeg", ".png", ".xml"}:
            raise ValueError(f"Weights/data cannot enter code-only package: {relative}")
    manifest = {p.as_posix(): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, PREFIX + relative.as_posix())
        archive.writestr(PREFIX + "STAGE2_PACKAGE_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC failed")
        if len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("Unexpected archive member count")
        for relative, digest in manifest.items():
            if hashlib.sha256(archive.read(PREFIX + relative)).hexdigest() != digest:
                raise RuntimeError(f"Archive digest mismatch: {relative}")
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "files": len(FILES)+1, "training_entrypoint": False,
        "contains_weights_or_data": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
