"""Build two geometry-only V12 zonglie probes; preserve every score and candidate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_v31_zonglie_box_extent import change_box


SOURCE = Path(r"D:\新建文件夹 (8)\semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip")
OUTPUT_DIR = Path(r"D:\新建文件夹 (8)")
TEST = ROOT / "data/semifinal/test"
REPORT = ROOT / "runs/semifinal/v31_zonglie_extent_20261003/submission_build.json"
VARIANTS = {
    "strong": (1.5, 0.85),
    "conservative": (1.25, 1.0),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source() -> list[dict]:
    with ZipFile(SOURCE) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("V12 source is not a valid single-JSON ZIP")
        rows = json.loads(archive.read("submission.json"))
    if not isinstance(rows, list) or len(rows) != 165421:
        raise RuntimeError("Unexpected V12 source count")
    return rows


def sizes() -> dict[str, tuple[int, int]]:
    files = [path for path in TEST.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(files) != 788 or len({path.name for path in files}) != 788:
        raise RuntimeError(f"Expected 788 unique Test images, got {len(files)}")
    output = {}
    for path in files:
        with Image.open(path) as image:
            output[path.name] = image.size
    return output


def main() -> None:
    baseline = read_source()
    image_sizes = sizes()
    if not {row["image_id"] for row in baseline} <= set(image_sizes):
        raise RuntimeError("Submission image_id is not in the frozen Test set")
    counts = Counter(row["category_name"] for row in baseline)
    report = {"source": str(SOURCE), "source_sha256": sha(SOURCE),
              "images_test": len(image_sizes), "rows": len(baseline), "classes": dict(counts), "variants": {}}
    for label, (height_scale, width_scale) in VARIANTS.items():
        output = OUTPUT_DIR / f"semifinal_v31_zonglie_extent_{label}_SUBMIT_ONLY.zip"
        if output.exists():
            raise FileExistsError(output)
        changed = []
        modified = 0
        for row in baseline:
            if row["category_name"] == "zonglie":
                item = change_box(row, image_sizes, height_scale, width_scale)
                modified += item["bbox"] != row["bbox"]
            else:
                item = row
            x1, y1, x2, y2 = item["bbox"]
            width, height = image_sizes[item["image_id"]]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise RuntimeError(f"Invalid box: {item}")
            changed.append(item)
        if len(changed) != len(baseline) or not modified:
            raise RuntimeError("Candidate identity was not preserved")
        encoded = (json.dumps(changed, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("submission.json", encoded)
        with ZipFile(output) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Output ZIP failed validation")
        report["variants"][label] = {"output": str(output), "sha256": sha(output),
                                      "height_scale": height_scale, "width_scale": width_scale,
                                      "bbox_changed": modified, "rows": len(changed),
                                      "single_root_json": True}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
