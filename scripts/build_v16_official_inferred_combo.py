"""Combine officially measured nonnegative V14 classes onto protected V12 JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


CLASSES = {"jieba", "mamianmakeng"}


def read_zip(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise ValueError(f"expected exactly one root submission.json: {path}")
        rows = json.loads(archive.read("submission.json"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid rows: {path}")
    return rows


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12", type=Path, required=True)
    parser.add_argument("--v14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v12, v14 = read_zip(args.v12), read_zip(args.v14)
    if len(v12) != len(v14):
        raise ValueError("row count mismatch")
    output = []
    changed = {name: 0 for name in CLASSES}
    for old, new in zip(v12, v14, strict=True):
        for key in ("image_id", "category_name", "bbox"):
            if old[key] != new[key]:
                raise ValueError(f"identity mismatch: {key}")
        item = dict(old)
        if item["category_name"] in CLASSES:
            item["score"] = new["score"]
            changed[item["category_name"]] += float(old["score"]) != float(new["score"])
        output.append(item)
    if any(count == 0 for count in changed.values()):
        raise ValueError(f"empty class change: {changed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    encoded = (json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", encoded)
    with zipfile.ZipFile(args.output) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("invalid output ZIP")
    print(json.dumps({
        "output": str(args.output), "sha256": digest(args.output), "rows": len(output),
        "changed_rows": changed, "v12_zip_sha256": digest(args.v12), "v14_zip_sha256": digest(args.v14),
        "membership_geometry_unchanged": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
