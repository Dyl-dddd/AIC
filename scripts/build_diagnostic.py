"""Create local review sheets; materialize a train=self-eval diagnostic only after review."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import Tile, clip_box_to_tile, xyxy_to_yolo
from steel_defect.governance import digest_file
from steel_defect.image_io import imread, imwrite
from steel_defect.voc import parse_voc


def make_sheets(items, folder, prefix):
    for page, start in enumerate(range(0, len(items), 12)):
        canvas = np.full((3 * 410, 4 * 384, 3), 245, np.uint8)
        for local, item in enumerate(items[start:start + 12]):
            image = imread(folder / item["image"], cv2.IMREAD_COLOR)
            preview = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
            for cls, box in item["annotations"]:
                x1, y1, x2, y2 = [round(v * 384 / 1024) for v in box]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(preview, str(cls), (max(0, x1), max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            x, y = local % 4 * 384, local // 4 * 410
            canvas[y:y + 384, x:x + 384] = preview
            cv2.putText(canvas, item["id"], (x + 5, y + 402), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
        imwrite(folder / f"{prefix}_{page:02d}.jpg", canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accept-review", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    if args.accept_review:
        review = json.loads(args.accept_review.read_text(encoding="utf-8"))
        candidate = json.loads((args.output / "candidates.json").read_text(encoding="utf-8"))
        if review["candidate_sha256"] != digest_file(args.output / "candidates.json"):
            raise ValueError("Review does not match candidates")
        if candidate["split_sha256"] != digest_file(args.manifest):
            raise ValueError("Candidate split changed")
        lookup = {i["id"]: i for i in candidate["items"]}
        chosen = [lookup[key] for key in review["accepted_ids"]]
        if len(chosen) != len(set(review["accepted_ids"])):
            raise ValueError("Duplicate selected sample")
        negatives = sum(not item["annotations"] for item in chosen)
        if negatives != 8 or not 32 <= len(chosen) - negatives <= 64:
            raise ValueError("Diagnostic needs 32-64 positives and exactly 8 negatives")
        images, labels = args.output / "images" / "diagnostic", args.output / "labels" / "diagnostic"
        if images.exists() or labels.exists():
            raise ValueError("Diagnostic already materialized")
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for item in chosen:
            original = source / item["source"]
            expected = manifest["records"][item["source"]]
            if item["source"] not in manifest["splits"]["train"] or digest_file(original) != expected["sha256"]:
                raise ValueError("Diagnostic source is not frozen training data")
            if digest_file(original.with_suffix(".xml")) != expected["xml_sha256"]:
                raise ValueError("Diagnostic source annotation changed")
            path = args.output / item["image"]
            if digest_file(path) != item["image_sha256"]:
                raise ValueError("Reviewed crop changed")
            shutil.copy2(path, images / f"{item['id']}.jpg")
            lines = [f"{cls} " + " ".join(f"{v:.8f}" for v in xyxy_to_yolo(box, 1024, 1024))
                     for cls, box in item["annotations"]]
            (labels / f"{item['id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        report = {"diagnostic_only": True, "train_equals_validation": True, "not_generalization": True,
                  "split_sha256": candidate["split_sha256"], "review": review,
                  "samples": len(chosen), "negative_samples": negatives,
                  "class_objects": dict(Counter(CLASS_NAMES[c] for i in chosen for c, _ in i["annotations"])), "items": chosen}
        (args.output / "diagnostic_manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        spec = {"path": args.output.resolve().as_posix(), "train": "images/diagnostic", "val": "images/diagnostic",
                "nc": 9, "names": dict(enumerate(CLASS_NAMES))}
        (args.output / "diagnostic.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k not in {"items", "review"}}, indent=2))
        return
    if args.output.exists() and any(p.is_file() for p in args.output.rglob("*")):
        raise ValueError("Review output already exists")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidates").mkdir(exist_ok=True)
    records = [parse_voc(source / path, (source / path).with_suffix(".xml")) for path in manifest["splits"]["train"]]
    rng = random.Random(42)
    rng.shuffle(records)
    selected, used = [], set()
    for class_id in range(9):
        options = []
        for record in records:
            anns = [a for a in record.annotations if a.class_id == class_id]
            if not anns:
                continue
            ann = max(anns, key=lambda a: (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]))
            x1, y1, x2, y2 = ann.box
            # Prefer isolated, visually resolvable defects; preserve long-box fragments.
            rank = (len(record.annotations), -min(x2 - x1, y2 - y1), record.image_path.name)
            options.append((rank, record, ann))
        count = 0
        for _, record, ann in sorted(options, key=lambda x: x[0]):
            if record.image_path in used:
                continue
            x1, y1, x2, y2 = ann.box
            x = max(0, min(record.width - 1024, round((x1 + x2) / 2 - 512)))
            y = max(0, min(record.height - 1024, round((y1 + y2) / 2 - 512)))
            selected.append((f"p{class_id}_{count}", record, x, y))
            used.add(record.image_path)
            count += 1
            if count == 4:
                break
        if count < 4:
            raise ValueError(f"Insufficient diagnostic positives for {class_id}")
    negatives = [r for r in records if not r.annotations][:50]
    selected += [(f"n{i:02d}", r, (r.width - 1024) // 2, (r.height - 1024) // 2) for i, r in enumerate(negatives)]
    items = []
    for key, record, x, y in selected:
        raw = imread(record.image_path, cv2.IMREAD_GRAYSCALE)
        tile = Tile(x, y, 1024, 1024, record.width, record.height)
        anns = []
        for ann in record.annotations:
            box = clip_box_to_tile(ann.box, tile, min_visibility=0)
            if box is not None:
                anns.append((ann.class_id, box))
        image = f"candidates/{key}.jpg"
        imwrite(args.output / image, raw[y:y + 1024, x:x + 1024], params=[cv2.IMWRITE_JPEG_QUALITY, 95])
        item = {"id": key, "image": image, "image_sha256": digest_file(args.output / image),
                "source": record.image_path.relative_to(source).as_posix(), "origin": [x, y], "annotations": anns}
        if not anns:
            preview = cv2.cvtColor(cv2.resize(raw, (512, 375), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(preview, (round(x / 8), round(y / 8)), (round((x + 1024) / 8), round((y + 1024) / 8)), (0, 255, 0), 1)
            full = np.full((512, 512, 3), 245, np.uint8)
            full[:375] = preview
            item["source_preview"] = f"candidates/{key}_source.jpg"
            imwrite(args.output / item["source_preview"], full)
        items.append(item)
    make_sheets([i for i in items if i["annotations"]], args.output, "positive_review")
    make_sheets([i for i in items if not i["annotations"]], args.output, "negative_review")
    make_sheets([{**i, "image": i["source_preview"]} for i in items if not i["annotations"]], args.output, "negative_source_review")
    (args.output / "candidates.json").write_text(json.dumps({"split_sha256": digest_file(args.manifest), "items": items}, indent=2), encoding="utf-8")
    print(f"Review candidates saved: {args.output}; no train YAML until accepted review")


if __name__ == "__main__":
    main()
