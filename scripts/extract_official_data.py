"""Validate and normalize the official train/test ZIP archives.

Archive entries are treated as untrusted data: files are selected by type and
stem, never extracted by their stored paths. The source ZIP files are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.classes import CLASS_NAMES, canonical_class


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def _safe_infos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = [info for info in archive.infolist() if not info.is_dir()]
    unsafe = []
    for info in infos:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        first = path.parts[0] if path.parts else ""
        if path.is_absolute() or ".." in path.parts or ":" in first or info.flag_bits & 1:
            unsafe.append(info.filename)
    if unsafe:
        raise ValueError(f"Unsafe or encrypted ZIP entries: {unsafe[:5]}")
    return infos


def _index_unique(
    infos: list[zipfile.ZipInfo], suffixes: set[str], kind: str
) -> dict[str, zipfile.ZipInfo]:
    selected: dict[str, zipfile.ZipInfo] = {}
    duplicates: list[str] = []
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.suffix.lower() not in suffixes:
            continue
        key = path.stem.casefold()
        if key in selected:
            duplicates.append(path.stem)
        else:
            selected[key] = info
    if duplicates:
        raise ValueError(f"Duplicate {kind} stems: {duplicates[:10]}")
    return selected


def _image_metadata(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int, str]:
    with archive.open(info) as handle:
        with Image.open(handle) as image:
            metadata = (image.width, image.height, image.mode)
            image.verify()
            return metadata


def _format_coordinate(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.6f}".rstrip("0").rstrip(".")


def _normalize_xml(
    payload: bytes,
    image_name: str,
    image_size: tuple[int, int],
    image_mode: str,
) -> tuple[bytes, Counter[str], Counter[str], int]:
    root = ET.fromstring(payload)
    repairs: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()

    filename = root.find("filename")
    if filename is None:
        filename = ET.SubElement(root, "filename")
        repairs["filename_inserted"] += 1
    elif (filename.text or "").strip() != image_name:
        declared = PurePosixPath((filename.text or "").strip().replace("\\", "/"))
        if declared.stem.casefold() == Path(image_name).stem.casefold():
            repairs["filename_normalized"] += 1
        else:
            repairs["filename_stem_corrected"] += 1
    filename.text = image_name

    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
        repairs["size_inserted"] += 1
    width, height = image_size
    if image_mode != "L":
        raise ValueError(f"Expected an 8-bit grayscale image, got mode={image_mode} for {image_name}")
    expected = {"width": width, "height": height, "depth": 1 if image_mode == "L" else 3}
    for field, value in expected.items():
        node = size.find(field)
        if node is None:
            node = ET.SubElement(size, field)
            repairs[f"{field}_inserted"] += 1
        else:
            try:
                matches = int(float(node.text or "0")) == value
            except ValueError:
                matches = False
            if not matches:
                repairs[f"{field}_corrected"] += 1
        node.text = str(value)

    objects = root.findall("object")
    for obj in objects:
        name_node = obj.find("name")
        raw_name = (name_node.text or "").strip() if name_node is not None else ""
        normalized = canonical_class(raw_name)
        if normalized not in CLASS_NAMES:
            raise ValueError(f"Unknown official class {raw_name!r} in {image_name}")
        if name_node is None:
            name_node = ET.SubElement(obj, "name")
            repairs["class_inserted"] += 1
        elif normalized != raw_name:
            repairs["class_normalized"] += 1
        name_node.text = normalized
        class_counts[normalized] += 1

        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {image_name}")
        values: dict[str, float] = {}
        for field in ("xmin", "ymin", "xmax", "ymax"):
            node = box.find(field)
            if node is None:
                raise ValueError(f"Missing {field} in {image_name}")
            values[field] = float(node.text or "nan")
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite box coordinate in {image_name}: {values}")
        clipped = {
            "xmin": min(max(values["xmin"], 0.0), float(width)),
            "ymin": min(max(values["ymin"], 0.0), float(height)),
            "xmax": min(max(values["xmax"], 0.0), float(width)),
            "ymax": min(max(values["ymax"], 0.0), float(height)),
        }
        if clipped != values:
            repairs["box_clipped"] += 1
        if clipped["xmax"] <= clipped["xmin"] or clipped["ymax"] <= clipped["ymin"]:
            raise ValueError(f"Degenerate box after clipping in {image_name}: {values}")
        for field, value in clipped.items():
            box.find(field).text = _format_coordinate(value)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), repairs, class_counts, len(objects)


def inspect_archives(
    train_zip: Path,
    test_zip: Path,
    destination: Path | None = None,
) -> dict:
    train_zip = train_zip.resolve()
    test_zip = test_zip.resolve()
    report: dict = {
        "sources": {
            "train_zip": str(train_zip),
            "test_zip": str(test_zip),
        },
        "source_archives_modified": False,
    }
    train_dir = destination / "train" if destination else None
    test_dir = destination / "test" if destination else None
    if destination:
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

    with zipfile.ZipFile(train_zip) as archive:
        infos = _safe_infos(archive)
        images = _index_unique(infos, IMAGE_SUFFIXES, "training image")
        xmls = _index_unique(infos, {".xml"}, "training XML")
        missing_xml = sorted(set(images) - set(xmls))
        orphan_xml = sorted(set(xmls) - set(images))
        if missing_xml or orphan_xml:
            raise ValueError(
                f"Training pairs disagree: missing_xml={len(missing_xml)}, orphan_xml={len(orphan_xml)}"
            )

        repairs: Counter[str] = Counter()
        classes: Counter[str] = Counter()
        train_dimensions: Counter[str] = Counter()
        train_modes: Counter[str] = Counter()
        empty_annotations = 0
        training_names: set[str] = set()
        for key in sorted(images):
            image_info = images[key]
            image_name = f"{PurePosixPath(image_info.filename).stem}.jpg"
            image_size = _image_metadata(archive, image_info)
            width, height, mode = image_size
            train_dimensions[f"{width}x{height}"] += 1
            train_modes[mode] += 1
            xml_payload, item_repairs, item_classes, object_count = _normalize_xml(
                archive.read(xmls[key]), image_name, (width, height), mode
            )
            repairs.update(item_repairs)
            classes.update(item_classes)
            empty_annotations += object_count == 0
            training_names.add(image_name.casefold())
            if destination:
                with archive.open(image_info) as source, (train_dir / image_name).open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                (train_dir / f"{Path(image_name).stem}.xml").write_bytes(xml_payload)

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
        test_dimensions: Counter[str] = Counter()
        test_modes: Counter[str] = Counter()
        test_names: set[str] = set()
        for key in sorted(images):
            image_info = images[key]
            image_name = f"{PurePosixPath(image_info.filename).stem}.jpg"
            width, height, mode = _image_metadata(archive, image_info)
            test_dimensions[f"{width}x{height}"] += 1
            test_modes[mode] += 1
            test_names.add(image_name.casefold())
            if destination:
                with archive.open(image_info) as source, (test_dir / image_name).open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)

    overlap = training_names & test_names
    if overlap:
        raise ValueError(f"Train/test filename overlap: {len(overlap)}")
    report["train"] = {
        "images": len(training_names),
        "xmls": len(training_names),
        "objects": sum(classes.values()),
        "empty_annotations": empty_annotations,
        "class_counts": {name: classes[name] for name in CLASS_NAMES},
        "dimensions": dict(train_dimensions),
        "modes": dict(train_modes),
        "repairs": dict(sorted(repairs.items())),
    }
    report["test"] = {
        "images": len(test_names),
        "dimensions": dict(test_dimensions),
        "modes": dict(test_modes),
    }
    report["train_test_filename_overlap"] = 0
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Safely normalize official competition ZIP files.")
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--test-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Output root containing train/ and test/")
    parser.add_argument("--verify-only", action="store_true", help="Audit without extracting images")
    parser.add_argument("--report", type=Path, help="Optional JSON audit report")
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
        report = inspect_archives(args.train_zip, args.test_zip, staging)
        if staging:
            (staging / "normalization_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            staging.rename(args.output.resolve())
    except Exception:
        if staging:
            print(f"Extraction stopped; partial files preserved for inspection at {staging}")
        raise

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
