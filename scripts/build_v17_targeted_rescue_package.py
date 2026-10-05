"""Build a code-only V17 cloud package with deterministic member checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "semifinal_v17_targeted_rescue_20261002"
MEMBERS = [
    "README_V17_TARGETED_RESCUE_4090.md",
    "requirements-v17.txt",
    "configs/train/semifinal_v17_targeted_rescue_4090.yaml",
    "scripts/run_v17_targeted_rescue_pipeline.py",
    "scripts/run_full_retrain_pipeline.py",
    "scripts/run_semifinal_stage1.py",
    "scripts/select_full_retrain_checkpoint.py",
    "scripts/train.py",
    "tests/test_v17_targeted_rescue_package.py",
    "steel_defect/__init__.py",
    "steel_defect/classes.py",
    "steel_defect/deblur.py",
    "steel_defect/focal_trainer.py",
    "steel_defect/geometry.py",
    "steel_defect/governance.py",
    "steel_defect/image_io.py",
    "steel_defect/inference.py",
    "steel_defect/metrics.py",
    "steel_defect/quality_trainer.py",
    "steel_defect/recall_booster.py",
    "steel_defect/runtime.py",
    "steel_defect/safe_r2.py",
    "steel_defect/training_audit.py",
    "steel_defect/voc.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {name: (ROOT / name).read_bytes() for name in MEMBERS}
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    files["V17_PACKAGE_MANIFEST.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in files.items():
            entry = zipfile.ZipInfo(f"{PACKAGE_NAME}/{name}")
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, content)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        for name, digest in manifest.items():
            actual = hashlib.sha256(archive.read(f"{PACKAGE_NAME}/{name}")).hexdigest()
            if actual != digest:
                raise ValueError(f"packaged member digest mismatch: {name}")
    print(json.dumps({"archive": str(args.output), "bytes": args.output.stat().st_size,
                      "members": len(files), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
