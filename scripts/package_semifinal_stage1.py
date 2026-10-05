"""Create an isolated, evaluation-only cloud upload package."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "updates/semifinal_stage1_eval_20260923.zip"
FILES = [Path("README_STAGE1_4090.md"), Path("scripts/run_semifinal_stage1.py"),
    Path("tests/test_stage1_geometry.py"),
    *sorted(p.relative_to(ROOT) for p in (ROOT / "steel_defect").glob("*.py"))]


def main():
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {DESTINATION}")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    manifest = {str(p.as_posix()): hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}
    with zipfile.ZipFile(DESTINATION, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in FILES:
            archive.write(ROOT / path, "semifinal_stage1_20260923/" + path.as_posix())
        archive.writestr("semifinal_stage1_20260923/STAGE1_PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(DESTINATION) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(FILES) + 1
        for relative, digest in manifest.items():
            assert hashlib.sha256(archive.read("semifinal_stage1_20260923/" + relative)).hexdigest() == digest
    print(json.dumps({"path": str(DESTINATION), "bytes": DESTINATION.stat().st_size,
        "sha256": hashlib.sha256(DESTINATION.read_bytes()).hexdigest(), "files": len(FILES)+1,
        "training_entrypoint": False, "includes_weights_or_data": False}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
