"""PASCAL VOC parsing and dataset discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .classes import CLASS_TO_ID, canonical_class

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Annotation:
    class_name: str
    class_id: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class VocRecord:
    image_path: Path
    xml_path: Path
    width: int
    height: int
    annotations: tuple[Annotation, ...]
    source_group: str | None = None

    @property
    def group_id(self) -> str:
        if self.source_group is not None:
            return self.source_group
        # Keep adjacent RawXX frames from the same steel coil in one split.
        competition_group = re.split(
            r"[-_]Raw\d+", self.image_path.stem, maxsplit=1, flags=re.IGNORECASE
        )[0]
        if competition_group != self.image_path.stem:
            return competition_group
        # GC10-DET: img_<camera>_<batch-or-coil>_<frame>.
        gc10 = re.match(r"^(img_\d+_.+)_\d+$", self.image_path.stem, flags=re.IGNORECASE)
        return gc10.group(1) if gc10 else self.image_path.stem


def discover_pairs(source: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    images = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    xml_by_key: dict[tuple[str, str], Path] = {}
    for xml in source.rglob("*.xml"):
        xml_by_key[(str(xml.parent).lower(), xml.stem.lower())] = xml

    pairs: list[tuple[Path, Path]] = []
    missing_xml: list[Path] = []
    used_xml: set[Path] = set()
    for image in images:
        direct = image.with_suffix(".xml")
        xml = direct if direct.exists() else xml_by_key.get((str(image.parent).lower(), image.stem.lower()))
        if xml is None:
            # Common VOC layout: JPEGImages/foo.jpg and Annotations/foo.xml.
            candidates = list(source.rglob(f"{image.stem}.xml"))
            xml = candidates[0] if len(candidates) == 1 else None
        if xml is None:
            missing_xml.append(image)
        else:
            pairs.append((image, xml))
            used_xml.add(xml.resolve())

    orphan_xml = [p for p in source.rglob("*.xml") if p.resolve() not in used_xml]
    return pairs, missing_xml, orphan_xml


def discover_class_names(xml_paths: list[Path]) -> list[str]:
    """Return sorted, exact class labels found in a collection of VOC XML files."""
    names: set[str] = set()
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            if name:
                names.add(name)
    def natural_key(value: str):
        return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]

    return sorted(names, key=natural_key)


def parse_voc(
    image_path: Path,
    xml_path: Path,
    strict: bool = True,
    class_to_id: dict[str, int] | None = None,
) -> VocRecord:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size>: {xml_path}")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    mapping = CLASS_TO_ID if class_to_id is None else class_to_id
    use_official_aliases = class_to_id is None
    annotations: list[Annotation] = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "")
        name = canonical_class(raw_name) if use_official_aliases else raw_name.strip()
        if name not in mapping:
            if strict:
                raise ValueError(f"Unknown class {raw_name!r} in {xml_path}")
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            if strict:
                raise ValueError(f"Missing <bndbox> in {xml_path}")
            continue
        x1 = max(0.0, min(float(bnd.findtext("xmin", "0")), width))
        y1 = max(0.0, min(float(bnd.findtext("ymin", "0")), height))
        x2 = max(0.0, min(float(bnd.findtext("xmax", "0")), width))
        y2 = max(0.0, min(float(bnd.findtext("ymax", "0")), height))
        if x2 <= x1 or y2 <= y1:
            if strict:
                raise ValueError(f"Invalid box {(x1, y1, x2, y2)} in {xml_path}")
            continue
        annotations.append(Annotation(name, mapping[name], (x1, y1, x2, y2)))

    return VocRecord(image_path, xml_path, width, height, tuple(annotations))
