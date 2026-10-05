"""Build a frozen v2 reference dataset without copying or changing source images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import base64
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steel_defect.classes import CLASS_NAMES
from steel_defect.governance import connected_groups, digest_file, validate_split_manifest, source_keys
from steel_defect.image_io import imread
from steel_defect.voc import discover_pairs, parse_voc, IMAGE_EXTENSIONS
from scripts.prepare_data import split_records


def image_fingerprint(task):
    path, root, kind = task
    image = imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    h, w = image.shape
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
    rel = path.relative_to(root).as_posix()
    return {"id": kind + ":" + rel, "kind": kind, "path": rel,
            "sha256": digest_file(path), "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "width": w, "height": h, "dhash": bits,
            "thumbnail": base64.b64encode(cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA).tobytes()).decode("ascii")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fingerprints", type=Path, help="Reuse fingerprints only after byte/XML verification")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; use a new version directory")
    args.output.mkdir(parents=True)
    pairs, missing, orphan = discover_pairs(args.source)
    if missing or orphan or not pairs:
        raise ValueError(f"Invalid pairs: missing={len(missing)}, orphan={len(orphan)}")
    records = [parse_voc(image, xml) for image, xml in pairs]
    test = sorted(p for p in args.test.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    tasks = [(r.image_path, args.source, "train") for r in records] + [(p, args.test, "test") for p in test]
    cv2.setNumThreads(1)
    entries = []
    if args.fingerprints:
        entries = json.loads(args.fingerprints.read_text(encoding="utf-8"))
        expected = {kind + ":" + p.relative_to(root).as_posix() for p, root, kind in tasks}
        if {e["id"] for e in entries} != expected or len(entries) != len(expected):
            raise ValueError("Fingerprint scope mismatch")
        for e in entries:
            root = args.source if e["kind"] == "train" else args.test
            if digest_file(root / e["path"]) != e["sha256"]:
                raise ValueError("Image changed since fingerprints")
            if e["kind"] == "train" and digest_file((root / e["path"]).with_suffix(".xml")) != e["xml_sha256"]:
                raise ValueError("Annotation changed since fingerprints")
        print("Verified cached image/XML fingerprints", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, entry in enumerate(pool.map(image_fingerprint, tasks), 1):
                entries.append(entry)
                if i % 100 == 0:
                    print(f"Fingerprints {i}/{len(tasks)}", flush=True)
    lookup = {"train:" + r.image_path.relative_to(args.source).as_posix(): r for r in records}
    for entry in entries:
        if entry["kind"] == "train":
            record = lookup[entry["id"]]
            if (entry["width"], entry["height"]) != (record.width, record.height):
                raise ValueError(f"XML/image size mismatch: {entry['id']}")
            entry["xml_sha256"] = digest_file(record.xml_path)
            entry["annotations"] = sorted([[a.class_name, *a.box] for a in record.annotations])
    (args.output / "fingerprints.json").write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    print("Building byte/pixel/source/perceptual connected groups", flush=True)
    groups, near = connected_groups(entries)
    by_pixels = defaultdict(list)
    for entry in entries:
        if entry["kind"] == "train":
            by_pixels[entry["pixel_sha256"]].append(entry)
    conflicts = [items for items in by_pixels.values() if len(items) > 1 and
                 len({json.dumps(item["annotations"]) for item in items}) > 1]
    quarantined = defaultdict(set)
    for entry in entries:
        if entry["kind"] == "test":
            quarantined[groups[entry["id"]]].add("test_connected_component")
        if entry["kind"] == "train":
            annotations = [tuple(a) for a in entry["annotations"]]
            if len(annotations) != len(set(annotations)):
                quarantined[groups[entry["id"]]].add("duplicate_gt_boxes_review_required")
    for items in conflicts:
        for item in items:
            quarantined[groups[item["id"]]].add("identical_pixels_conflicting_annotations")
    eligible = [replace(r, source_group=groups[key]) for key, r in lookup.items()
                if groups[key] not in quarantined]
    (args.output / "grouping_status.json").write_text(json.dumps({
        "eligible": len(eligible), "source_images": len(records),
        "excluded_by_reason": dict(Counter(reason for key in lookup for reason in quarantined.get(groups[key], []))),
        "near_pairs": len(near), "group_sizes": dict(Counter(groups[key] for key in lookup))}, indent=2), encoding="utf-8")
    if len(eligible) < len(records) * 0.5:
        raise ValueError("Conservative grouping excluded over half the training set; review before splitting")
    holdout = split_records(eligible, .10, args.seed, args.trials)
    dev = split_records(holdout["train"], .15 / .90, args.seed + 1009, args.trials)
    splits = {"train": dev["train"], "dev": dev["val"], "final": holdout["val"]}
    train_entries = {e["path"]: e for e in entries if e["kind"] == "train"}
    manifest = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()), "test_source": str(args.test.resolve()),
        "seed": args.seed, "classes": CLASS_NAMES,
        "roles": {"train": "train", "dev": "dev", "final": "final_holdout"},
        "splits": {key: [r.image_path.relative_to(args.source).as_posix() for r in items] for key, items in splits.items()},
        "groups": {key: sorted({r.group_id for r in items}) for key, items in splits.items()},
        "records": {path: {**entry, "group": groups[entry["id"]]} for path, entry in train_entries.items()},
        "grouping_policy": "Raw + candidate C + byte/pixel SHA + verified near pairs (dHash<=4, thumbnail corr>=.9995, RMSE<=1, std>=4); C semantics unverified",
        "near_duplicate_limitations": "dHash is conservative isolation, not proof of identity or exhaustive duplicate detection",
    }
    validate_split_manifest(manifest)
    exclusions = [{"path": e["path"], "group": groups[e["id"]], "reasons": sorted(quarantined[groups[e["id"]]])}
                  for e in entries if e["kind"] == "train" and groups[e["id"]] in quarantined]
    counts = {key: dict(Counter(a.class_name for r in items for a in r.annotations)) for key, items in splits.items()}
    if any(set(counts[key]) != set(CLASS_NAMES) for key in splits):
        raise ValueError("At least one class missing from a split; review group feasibility")
    report = {"status": "frozen_split_created_label_review_pending", "source_images": len(records),
              "test_images": len(test), "excluded_training_images": len(exclusions),
              "split_images": {key: len(items) for key, items in splits.items()},
              "class_objects": counts, "conflicting_pixel_groups": len(conflicts),
              "near_pairs": len(near), "source_unchanged": True,
              "limitations": "Label review and learning gate still required; no claim of exhaustive near-duplicate detection"}
    test_keys = {key for e in entries if e["kind"] == "test" for key in source_keys(e["path"])}
    prefix_overlap = [e["path"] for e in entries if e["kind"] == "train" and test_keys.intersection(source_keys(e["path"]))]
    report["train_test_source_prefix_overlap_images"] = len(prefix_overlap)
    report["test_link_policy"] = "Exclude training components connected by byte/pixel/verified-near content; prefix-only overlap is a reported limitation, not proven duplicate"
    outputs = {"splits.json": manifest, "fingerprints.json": entries, "quarantine.json": exclusions,
               "conflicts.json": conflicts, "near_pairs.json": near, "audit_summary.json": report,
               "test_prefix_overlap.json": prefix_overlap}
    for name, payload in outputs.items():
        (args.output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
