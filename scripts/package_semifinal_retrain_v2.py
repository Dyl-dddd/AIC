"""Build and verify the semifinal v2 code-only upload archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "updates" / "semifinal_retrain_v2_code_only_20260923.zip"
FILES = [
    Path("README_SEMIFINAL_RETRAIN_V2.md"),
    Path("requirements.txt"),
    Path("configs/data/semifinal_grid_v1.yaml"),
    Path("configs/train/semifinal_retrain_v2_4090.yaml"),
    Path("configs/yolo11s-p2.yaml"),
    Path("data/official_v2_20260831b/splits.json"),
    Path("scripts/prepare_data.py"),
    Path("scripts/train.py"),
    Path("scripts/preflight_semifinal_retrain_v2.py"),
    Path("scripts/select_semifinal_checkpoint_v2.py"),
    *sorted(path.relative_to(ROOT) for path in (ROOT / "steel_defect").glob("*.py")),
]


def main() -> None:
    if len(set(FILES)) != len(FILES):
        raise ValueError("Duplicate archive member")
    for relative in FILES:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
        if relative.suffix.lower() in {".pt", ".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Model weights and image data must not enter this code-only archive: {relative}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    contents = {}
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in FILES:
            source = ROOT / relative
            archive.write(source, relative.as_posix())
            contents[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(contents, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(OUTPUT) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC test failed")
        if len(archive.namelist()) != len(FILES) + 1:
            raise RuntimeError("Archive member count mismatch")
    print(json.dumps({
        "archive": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "files": len(FILES),
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "contains_weight": False,
        "contains_images": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
