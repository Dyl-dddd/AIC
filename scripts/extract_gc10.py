"""Normalize the public GC10-DET ZIP into unique VOC image/XML pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile


LABEL_ALIASES = {"10_yaozhed": "10_yaozhe"}
DROP_LABELS = {"d"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    report: dict = {"source_zip": str(args.zip.resolve())}
    with zipfile.ZipFile(args.zip) as archive:
        unsafe = [
            info.filename for info in archive.infolist()
            if PurePosixPath(info.filename).is_absolute() or ".." in PurePosixPath(info.filename).parts
        ]
        if unsafe:
            raise ValueError(f"Unsafe ZIP entries: {unsafe[:5]}")
        images: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        xmls: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            suffix = PurePosixPath(info.filename).suffix.lower()
            if suffix == ".jpg":
                images[PurePosixPath(info.filename).stem].append(info)
            elif suffix == ".xml":
                xmls[PurePosixPath(info.filename).stem] = info

        missing_xml = sorted(set(images) - set(xmls))
        duplicate_stems = sorted(stem for stem, entries in images.items() if len(entries) > 1)
        class_counts: Counter[str] = Counter()
        rewritten_labels: Counter[str] = Counter()
        dropped_objects: Counter[str] = Counter()
        extracted = 0
        for stem in sorted(set(images) & set(xmls)):
            entries = images[stem]
            # GC10 duplicates are the same decoded pixels stored with different JPEG metadata.
            sizes = {entry.file_size for entry in entries}
            if len(sizes) != 1:
                raise ValueError(f"Duplicate image sizes disagree for {stem}: {sizes}")
            (args.output / f"{stem}.jpg").write_bytes(archive.read(entries[0]))

            root = ET.fromstring(archive.read(xmls[stem]))
            for obj in list(root.findall("object")):
                name_node = obj.find("name")
                raw_name = (name_node.text or "").strip() if name_node is not None else ""
                if raw_name in DROP_LABELS:
                    root.remove(obj)
                    dropped_objects[raw_name] += 1
                    continue
                normalized = LABEL_ALIASES.get(raw_name, raw_name)
                if normalized != raw_name:
                    rewritten_labels[f"{raw_name}->{normalized}"] += 1
                if name_node is None:
                    raise ValueError(f"Missing object name in {stem}")
                name_node.text = normalized
                class_counts[normalized] += 1
            ET.ElementTree(root).write(
                args.output / f"{stem}.xml", encoding="utf-8", xml_declaration=True
            )
            extracted += 1

    report.update(
        {
            "paired_images": extracted,
            "duplicate_stems_deduplicated": duplicate_stems,
            "missing_xml_images_skipped": missing_xml,
            "rewritten_labels": dict(rewritten_labels),
            "dropped_objects": dict(dropped_objects),
            "class_counts": dict(sorted(class_counts.items())),
        }
    )
    (args.output / "normalization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
