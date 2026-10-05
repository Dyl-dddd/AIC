"""Safely validate and extract an image-only competition test ZIP.

Archive paths are never trusted. Images are indexed by their unique stem and
written with normalized ``.jpg`` names into a new directory. The source ZIP is
opened read-only and the destination is staged before an atomic rename.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_official_data import IMAGE_SUFFIXES, _image_metadata, _index_unique, _safe_infos


def inspect_test_archive(test_zip: Path, destination: Path | None = None) -> dict:
    test_zip = test_zip.resolve()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    names: set[str] = set()

    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(test_zip) as archive:
        infos = _safe_infos(archive)
        images = _index_unique(infos, IMAGE_SUFFIXES, "test image")
        non_images = [
            info.filename
            for info in infos
            if PurePosixPath(info.filename).suffix.lower() not in IMAGE_SUFFIXES
        ]
        if non_images:
            raise ValueError(f"Unexpected non-image test entries: {non_images[:10]}")

        for key in sorted(images):
            image_info = images[key]
            image_name = f"{PurePosixPath(image_info.filename).stem}.jpg"
            width, height, mode = _image_metadata(archive, image_info)
            dimensions[f"{width}x{height}"] += 1
            modes[mode] += 1
            names.add(image_name.casefold())
            if destination is not None:
                with archive.open(image_info) as source, (destination / image_name).open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)

    return {
        "source_zip": str(test_zip),
        "source_archive_modified": False,
        "images": len(names),
        "dimensions": dict(dimensions),
        "modes": dict(modes),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Safely validate/extract an image-only test ZIP.")
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New output directory containing normalized JPGs")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    if not args.verify_only and args.output is None:
        parser.error("--output is required unless --verify-only is set")
    if args.output and args.output.exists():
        raise SystemExit(f"Output already exists; refusing to overwrite: {args.output}")

    staging = None
    if not args.verify_only:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.with_name(f"{output.name}.staging")
        if staging.exists():
            raise SystemExit(f"Staging path already exists; inspect it before retrying: {staging}")
        staging.mkdir()

    try:
        report = inspect_test_archive(args.test_zip, staging)
        if staging is not None:
            (staging / "archive_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            staging.rename(args.output.resolve())
    except Exception:
        if staging is not None:
            print(f"Extraction stopped; partial files preserved for inspection at {staging}")
        raise

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
