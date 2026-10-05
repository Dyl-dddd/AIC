"""Create and verify the isolated stage-3 checkpoint-soup archive."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_stage3_soup_20260924.zip"
PREFIX = "semifinal_stage3_20260924/"
FILES = [
    Path("README_STAGE3_4090.md"),
    Path("scripts/build_semifinal_soups.py"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("scripts/run_semifinal_stage3.py"),
    Path("tests/test_stage3_soup.py"),
    *sorted(path.relative_to(ROOT) for path in (ROOT / "steel_defect").glob("*.py")),
]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite release: {OUTPUT}")
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
        archive.writestr(PREFIX + "STAGE3_PACKAGE_MANIFEST.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC failed")
        for relative, digest in manifest.items():
            if hashlib.sha256(archive.read(PREFIX + relative)).hexdigest() != digest:
                raise RuntimeError(f"archive digest mismatch: {relative}")
    print(json.dumps({"archive": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), "files": len(FILES) + 1,
        "training_entrypoint": False, "contains_weights_or_data": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

