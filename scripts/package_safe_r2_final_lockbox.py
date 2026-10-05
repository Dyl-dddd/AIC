"""Build and verify the fail-closed SAFE-R2 final-lockbox code package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/safe_r2_final_lockbox_20260928.zip"
PREFIX = "safe_r2_final_lockbox_20260928/"
FILES = [
    Path("README_SAFE_R2_FINAL_LOCKBOX_4090.md"),
    Path("configs/inference/safe_r2_frozen_profile.json"),
    Path("scripts/analyze_final_submission_postprocess.py"),
    Path("scripts/analyze_source_aware_fusion.py"),
    Path("scripts/run_safe_r2_final_lockbox_pipeline.py"),
    Path("scripts/run_semifinal_stage1.py"),
    Path("tests/test_safe_r2.py"),
    Path("tests/test_safe_r2_lockbox.py"),
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
            raise ValueError(f"weights or data cannot enter code-only package: {relative}")
    manifest = {
        relative.as_posix(): hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in FILES
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, PREFIX + relative.as_posix())
        archive.writestr(
            PREFIX + "SAFE_R2_FINAL_LOCKBOX_PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("archive integrity check failed")
        names = archive.namelist()
        if any(Path(name).suffix.lower() in {".pt", ".jpg", ".jpeg", ".png", ".xml"} for name in names):
            raise RuntimeError("weights or data entered code-only package")
        if any(Path(name).name.lower() == "submission.json" for name in names):
            raise RuntimeError("leaderboard submission artifact entered lockbox package")
        for relative, expected in manifest.items():
            payload = archive.read(PREFIX + relative)
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError(f"archive digest mismatch: {relative}")
    print(json.dumps({
        "archive": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "files": len(FILES) + 1,
        "contains_weights_images_or_submission": False,
        "profile_deployment_allowed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
