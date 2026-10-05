"""Build and verify the code-only Ensemble V7 TTA release package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates/semifinal_ensemble_v7_tta_20260927.zip"
PREFIX = "semifinal_ensemble_v7_tta_20260927/"
FILES = [
    Path("README_ENSEMBLE_V7_TTA_4090.md"),
    Path("configs/inference/ensemble_v7_tta_primary.json"),
    Path("configs/inference/ensemble_v7_tta_conservative.json"),
    Path("scripts/analyze_final_submission_postprocess.py"),
    Path("scripts/analyze_source_aware_fusion.py"),
    Path("scripts/build_final_submission_from_cache.py"),
    Path("scripts/build_source_aware_submission_from_caches.py"),
    Path("scripts/infer.py"),
    Path("scripts/run_ensemble_v6_consensus_pipeline.py"),
    Path("scripts/run_ensemble_v7_tta_submission_pipeline.py"),
    Path("scripts/run_final_submission_pipeline.py"),
    Path("scripts/validate_large_submission.py"),
    Path("tests/test_ensemble_v6.py"),
    Path("tests/test_ensemble_v7_tta.py"),
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
            PREFIX + "ENSEMBLE_V7_TTA_PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("archive integrity check failed")
        for relative, expected in manifest.items():
            if hashlib.sha256(archive.read(PREFIX + relative)).hexdigest() != expected:
                raise RuntimeError(f"archive digest mismatch: {relative}")
    print(json.dumps({
        "archive": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "files": len(FILES) + 1,
        "contains_weights_or_images": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
