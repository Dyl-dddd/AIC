from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
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
from steel_defect.deblur import adaptive_deblur, apply_clahe
from steel_defect.geometry import clip_box_to_tile, generate_grid_tiles, generate_tiles, xyxy_to_yolo
from steel_defect.image_io import imread, imwrite
from steel_defect.runtime import apply_profile_defaults
from steel_defect.voc import VocRecord, discover_class_names, discover_pairs, parse_voc
from steel_defect.governance import frozen_records, digest_file


def stable_fraction(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_records(
    records: list[VocRecord],
    val_ratio: float,
    seed: int,
    trials: int = 2000,
) -> dict[str, list[VocRecord]]:
    """Group-aware approximate stratification using deterministic random search."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    groups: dict[str, list[VocRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("At least two independent groups are required for a leakage-safe train/val split")

    class_count = max((ann.class_id for record in records for ann in record.annotations), default=-1) + 1
    group_vectors: dict[str, np.ndarray] = {}
    group_sizes: dict[str, int] = {}
    for group_id, items in groups.items():
        vector = np.zeros(class_count, dtype=np.float64)
        for record in items:
            for ann in record.annotations:
                vector[ann.class_id] += 1.0
        group_vectors[group_id] = vector
        group_sizes[group_id] = len(items)

    total_vector = sum(group_vectors.values(), np.zeros(class_count, dtype=np.float64))
    target_vector = total_vector * val_ratio
    target_images = len(records) * val_ratio
    rng = random.Random(seed)
    best_score = float("inf")
    best_val: set[str] | None = None
    probability = min(0.95, max(0.05, val_ratio))
    for trial in range(max(1, trials)):
        if trial == 0:
            candidate = {group_id for group_id in group_ids if stable_fraction(group_id, seed) < val_ratio}
        else:
            candidate = {group_id for group_id in group_ids if rng.random() < probability}
        if not candidate or len(candidate) == len(group_ids):
            continue
        val_vector = sum((group_vectors[g] for g in candidate), np.zeros(class_count, dtype=np.float64))
        val_images = sum(group_sizes[g] for g in candidate)
        class_error = (
            float(np.mean(((val_vector - target_vector) / np.maximum(target_vector, 1.0)) ** 2))
            if class_count
            else 0.0
        )
        size_error = ((val_images - target_images) / max(target_images, 1.0)) ** 2
        train_vector = total_vector - val_vector
        splittable = sum((vector > 0).astype(np.int64) for vector in group_vectors.values()) >= 2
        missing_penalty = 4.0 * float(np.sum(splittable & ((val_vector == 0) | (train_vector == 0))))
        score = float(class_error + 0.5 * size_error + missing_penalty)
        if score < best_score:
            best_score = score
            best_val = candidate

    if best_val is None:
        best_val = {group_ids[0]}
    return {
        "train": [record for record in records if record.group_id not in best_val],
        "val": [record for record in records if record.group_id in best_val],
    }


def split_train_calibration_val(
    records: list[VocRecord],
    val_ratio: float,
    calibration_ratio: float,
    seed: int,
    trials: int,
) -> dict[str, list[VocRecord]]:
    """Create disjoint grouped holdout and threshold-calibration splits."""
    if calibration_ratio < 0 or val_ratio + calibration_ratio >= 1.0:
        raise ValueError("calibration_ratio must be non-negative and val+calibration must be below 1")
    primary = split_records(records, val_ratio, seed, trials=trials)
    if calibration_ratio == 0:
        return primary
    relative_ratio = calibration_ratio / (1.0 - val_ratio)
    secondary = split_records(primary["train"], relative_ratio, seed + 1009, trials=trials)
    return {
        "train": secondary["train"],
        "calibration": secondary["val"],
        "val": primary["val"],
    }


def resolve_classes(spec: str, xml_paths: list[Path]) -> list[str]:
    if spec == "official":
        return list(CLASS_NAMES)
    if spec == "auto":
        names = discover_class_names(xml_paths)
        if not names:
            raise ValueError("No <name> labels found while auto-discovering classes")
        return names
    path = Path(spec)
    if path.is_file():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = payload.get("names", payload) if isinstance(payload, dict) else payload
        if isinstance(names, dict):
            names = [names[index] for index in sorted(names, key=lambda value: int(value))]
        if not isinstance(names, list):
            raise ValueError(f"Class file must contain a list or names mapping: {path}")
        return [str(name) for name in names]
    names = [name.strip() for name in spec.split(",") if name.strip()]
    if not names:
        raise ValueError("--classes must be official, auto, a YAML file, or a comma-separated list")
    return names


def save_tile(
    image,
    record: VocRecord,
    tile,
    annotations,
    output: Path,
    split: str,
    deblur: bool,
    clahe: bool,
    deblur_threshold: float,
    view_tag: str | None = None,
) -> dict:
    crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
    blur_meta = {"applied": False, "blur_score": None, "strength": 0.0}
    if deblur:
        result = adaptive_deblur(crop, threshold=deblur_threshold)
        crop = result.image
        blur_meta = {
            "applied": result.applied,
            "blur_score": round(result.blur_score, 4),
            "strength": result.strength,
        }
    if clahe:
        crop = apply_clahe(crop)

    tag = f"__{view_tag}" if view_tag else ""
    stem = f"{record.image_path.stem}{tag}__x{tile.x:05d}_y{tile.y:05d}"
    image_out = output / "images" / split / f"{stem}.jpg"
    label_out = output / "labels" / split / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite(image_out, crop, params=[cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Failed to write {image_out}")
    lines = []
    for class_id, local_box in annotations:
        cx, cy, w, h = xyxy_to_yolo(local_box, tile.width, tile.height)
        lines.append(f"{class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "tile": stem,
        "source_image": record.image_path.name,
        "split": split,
        "origin": [tile.x, tile.y],
        "size": [tile.width, tile.height],
        "objects": len(annotations),
        "class_ids": sorted({class_id for class_id, _ in annotations}),
        "is_repeat": False,
        "is_global": False,
        "deblur": blur_meta,
    }


def save_global_view(
    image,
    record: VocRecord,
    output: Path,
    split: str,
    global_size: int,
    deblur: bool,
    clahe: bool,
    deblur_threshold: float,
) -> dict:
    scale = min(1.0, global_size / max(record.width, record.height))
    width = max(1, int(round(record.width * scale)))
    height = max(1, int(round(record.height * scale)))
    preview = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    blur_meta = {"applied": False, "blur_score": None, "strength": 0.0}
    if deblur:
        result = adaptive_deblur(preview, threshold=deblur_threshold)
        preview = result.image
        blur_meta = {
            "applied": result.applied,
            "blur_score": round(result.blur_score, 4),
            "strength": result.strength,
        }
    if clahe:
        preview = apply_clahe(preview)
    stem = f"{record.image_path.stem}__global"
    image_out = output / "images" / split / f"{stem}.jpg"
    label_out = output / "labels" / split / f"{stem}.txt"
    if not imwrite(image_out, preview, params=[cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Failed to write {image_out}")
    lines = []
    for ann in record.annotations:
        cx, cy, w, h = xyxy_to_yolo(ann.box, record.width, record.height)
        lines.append(f"{ann.class_id} {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}")
    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "tile": stem,
        "source_image": record.image_path.name,
        "split": split,
        "origin": [0, 0],
        "size": [width, height],
        "objects": len(record.annotations),
        "class_ids": sorted({ann.class_id for ann in record.annotations}),
        "is_repeat": False,
        "is_global": True,
        "deblur": blur_meta,
    }


def oversample_rare_tiles(metadata: list[dict], output: Path, repeat_cap: float) -> list[dict]:
    """Repeat training tiles using inverse-square-root class frequency."""
    if repeat_cap <= 1.0:
        return metadata
    presence: Counter[int] = Counter()
    for item in metadata:
        if item["split"] == "train" and not item.get("is_global", False):
            presence.update(item["class_ids"])
    if not presence:
        return metadata
    majority = max(presence.values())
    repeats: list[dict] = []
    for item in metadata:
        if item["split"] != "train" or item.get("is_global", False) or not item["class_ids"]:
            continue
        factor = max((majority / presence[class_id]) ** 0.5 for class_id in item["class_ids"])
        copies = max(0, int(round(min(repeat_cap, factor))) - 1)
        source_image = output / "images" / "train" / f"{item['tile']}.jpg"
        source_label = output / "labels" / "train" / f"{item['tile']}.txt"
        for repeat_index in range(1, copies + 1):
            repeated = dict(item)
            repeated["tile"] = f"{item['tile']}__rep{repeat_index}"
            repeated["is_repeat"] = True
            repeated["repeat_of"] = item["tile"]
            shutil.copy2(source_image, output / "images" / "train" / f"{repeated['tile']}.jpg")
            shutil.copy2(source_label, output / "labels" / "train" / f"{repeated['tile']}.txt")
            repeats.append(repeated)
    return metadata + repeats


def repeat_global_train_views(metadata: list[dict], output: Path, repeat_factor: int) -> list[dict]:
    """Repeat only training global views to match the semifinal image-size mixture."""
    if repeat_factor < 1:
        raise ValueError("global_repeat must be at least 1")
    if repeat_factor == 1:
        return metadata
    repeats: list[dict] = []
    for item in metadata:
        if item["split"] != "train" or not item.get("is_global", False) or item.get("is_repeat", False):
            continue
        source_image = output / "images" / "train" / f"{item['tile']}.jpg"
        source_label = output / "labels" / "train" / f"{item['tile']}.txt"
        for repeat_index in range(1, repeat_factor):
            repeated = dict(item)
            repeated["tile"] = f"{item['tile']}__globalrep{repeat_index}"
            repeated["is_repeat"] = True
            repeated["repeat_of"] = item["tile"]
            repeated["repeat_kind"] = "global_balance"
            shutil.copy2(source_image, output / "images" / "train" / f"{repeated['tile']}.jpg")
            shutil.copy2(source_label, output / "labels" / "train" / f"{repeated['tile']}.txt")
            repeats.append(repeated)
    return metadata + repeats


def process_record(record: VocRecord, split: str, args, class_names: list[str]) -> tuple[list[dict], Counter[str]]:
    image = imread(record.image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Cannot read {record.image_path}")
    actual_h, actual_w = image.shape[:2]
    if (actual_w, actual_h) != (record.width, record.height):
        raise ValueError(
            f"XML/image size mismatch for {record.image_path}: "
            f"XML={record.width}x{record.height}, image={actual_w}x{actual_h}"
        )
    metadata = []
    counts: Counter[str] = Counter()
    if args.view_mode != "global":
        tile_layout = getattr(args, "tile_layout", "sliding")
        tile_groups = []
        if tile_layout in {"sliding", "both"}:
            tile_groups.append((
                "sliding",
                generate_tiles(record.width, record.height, args.tile_size, args.overlap),
            ))
        if tile_layout in {"semifinal_grid", "both"}:
            tile_groups.append((
                "semifinal_grid",
                generate_grid_tiles(
                    record.width, record.height,
                    getattr(args, "tile_width", 1387), getattr(args, "tile_height", 1516),
                    getattr(args, "grid_rows", 2), getattr(args, "grid_cols", 3),
                ),
            ))
        for layout_name, tiles in tile_groups:
            positive = []
            negative = []
            for tile in tiles:
                tile_annotations = []
                omitted_fragment = False
                for ann in record.annotations:
                    local = clip_box_to_tile(ann.box, tile, min_visibility=args.min_visibility)
                    if local is not None:
                        tile_annotations.append((ann.class_id, local))
                    else:
                        x1, y1, x2, y2 = ann.box
                        if min(x2, tile.x + tile.width) > max(x1, tile.x) and min(y2, tile.y + tile.height) > max(y1, tile.y):
                            omitted_fragment = True
                # A rejected visible fragment is not clean background. Skip the
                # view instead of teaching it as an unlabeled negative.
                if omitted_fragment:
                    continue
                (positive if tile_annotations else negative).append((tile, tile_annotations))
            record_seed = int(hashlib.sha256(
                f"{args.seed}:{layout_name}:{record.image_path.name}".encode()
            ).hexdigest()[:16], 16)
            rng = random.Random(record_seed)
            max_negative = max(1, int(round(max(1, len(positive)) * args.negative_ratio)))
            rng.shuffle(negative)
            selected = positive + negative[:max_negative]
            selected.sort(key=lambda item: (item[0].y, item[0].x))
            view_tag = None if tile_layout == "sliding" else (
                "grid2x3" if layout_name == "semifinal_grid" else f"slide{args.tile_size}"
            )
            for tile, annotations in selected:
                metadata.append(
                    save_tile(
                        image, record, tile, annotations, args.output, split, args.deblur, args.clahe,
                        args.deblur_threshold, view_tag=view_tag,
                    )
                )
                for class_id, _ in annotations:
                    counts[class_names[class_id]] += 1
    if args.view_mode in {"global", "both"}:
        metadata.append(
            save_global_view(
                image, record, args.output, split, args.global_size, args.deblur, args.clahe,
                args.deblur_threshold,
            )
        )
        if args.view_mode == "global":
            for annotation in record.annotations:
                counts[class_names[annotation.class_id]] += 1
    return metadata, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VOC images to overlapping YOLO tiles.")
    parser.add_argument("--config", type=Path, help="YAML data-preparation profile")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen-splits", type=Path, help="Frozen v2 source manifest; final images are never prepared for training")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument(
        "--tile-layout", choices=("sliding", "semifinal_grid", "both"), default="sliding",
        help="Square sliding tiles, the observed semifinal 2x3 crop grid, or both",
    )
    parser.add_argument("--tile-width", type=int, default=1387)
    parser.add_argument("--tile-height", type=int, default=1516)
    parser.add_argument("--grid-rows", type=int, default=2)
    parser.add_argument("--grid-cols", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument(
        "--calibration-ratio", type=float, default=0.0,
        help="Independent grouped split for threshold/postprocess tuning; 0 disables",
    )
    parser.add_argument("--split-trials", type=int, default=2000)
    parser.add_argument(
        "--classes",
        default="official",
        help="official, auto, YAML path, or comma-separated labels (use auto for GC10-DET)",
    )
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument(
        "--rare-repeat-cap",
        type=float,
        default=1.0,
        help="Maximum inverse-frequency repeat factor for rare-class train tiles; 1 disables it",
    )
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="Parallel image preprocessing threads; 0 disables")
    deblur_group = parser.add_mutually_exclusive_group()
    deblur_group.add_argument(
        "--deblur",
        dest="deblur",
        action="store_true",
        help="Enable clarity-gated deblurring (experimental; disabled by default)",
    )
    deblur_group.add_argument(
        "--no-deblur",
        dest="deblur",
        action="store_false",
        help="Disable clarity-gated deblurring for an A/B baseline",
    )
    parser.set_defaults(deblur=False)
    parser.add_argument("--deblur-threshold", type=float, default=85.0)
    parser.add_argument("--clahe", action="store_true")
    global_group = parser.add_mutually_exclusive_group()
    global_group.add_argument("--global-context", dest="global_context", action="store_true")
    global_group.add_argument("--no-global-context", dest="global_context", action="store_false")
    parser.set_defaults(global_context=False)
    parser.add_argument(
        "--view-mode", choices=("tiles", "global", "both"), default=None,
        help="Training views; supersedes the legacy --global-context switch",
    )
    parser.add_argument("--global-size", type=int, default=1024)
    parser.add_argument(
        "--global-repeat", type=int, default=1,
        help="Total copies of each train global view; dev is never repeated",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = apply_profile_defaults(parser, "prepare")
    if args.source is None or args.output is None:
        parser.error("--source and --output are required (directly or through --config)")
    args.source = Path(args.source)
    args.output = Path(args.output)
    if args.view_mode is None:
        args.view_mode = "both" if args.global_context else "tiles"

    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output is not empty: {args.output}. Use --overwrite to replace it.")
        resolved = args.output.resolve()
        source_resolved = args.source.resolve()
        if (
            resolved == source_resolved
            or source_resolved in resolved.parents
            or resolved in source_resolved.parents
        ):
            raise SystemExit("Refusing an overwrite relationship between source and output directories.")
        shutil.rmtree(resolved)

    pairs, missing_xml, _ = discover_pairs(args.source)
    if missing_xml:
        raise SystemExit(f"{len(missing_xml)} images have no XML; run analyze_dataset.py first.")
    basename_counts = Counter(image.name for image, _ in pairs)
    duplicate_names = sorted(name for name, count in basename_counts.items() if count > 1)
    if duplicate_names:
        raise SystemExit(
            "Duplicate image basenames would overwrite generated tiles; rename them first: "
            + ", ".join(duplicate_names[:10])
        )
    class_names = resolve_classes(args.classes, [xml for _, xml in pairs])
    class_to_id = {name: index for index, name in enumerate(class_names)}
    custom_mapping = None if args.classes == "official" else class_to_id
    records = [parse_voc(image, xml, strict=True, class_to_id=custom_mapping) for image, xml in pairs]
    if not records:
        raise SystemExit("No valid image/XML pairs found.")

    frozen = None
    if args.frozen_splits:
        frozen = json.loads(Path(args.frozen_splits).read_text(encoding="utf-8"))
        if class_names != frozen["classes"]:
            raise ValueError("Frozen split classes mismatch")
        all_splits = frozen_records(records, args.source, frozen)
        splits = {key: value for key, value in all_splits.items()
                  if frozen["roles"][key] != "final_holdout"}
    else:
        splits = split_train_calibration_val(
            records, args.val_ratio, args.calibration_ratio, args.seed, trials=args.split_trials
        )
    metadata: list[dict] = []
    class_counts: Counter[str] = Counter()

    for split, split_records_list in splits.items():
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        if args.workers > 1:
            cv2.setNumThreads(1)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                processed = executor.map(
                    lambda record: process_record(record, split, args, class_names), split_records_list
                )
                for items, counts in processed:
                    metadata.extend(items)
                    class_counts.update(counts)
        else:
            for record in split_records_list:
                items, counts = process_record(record, split, args, class_names)
                metadata.extend(items)
                class_counts.update(counts)

    metadata = oversample_rare_tiles(metadata, args.output, args.rare_repeat_cap)
    metadata = repeat_global_train_views(metadata, args.output, args.global_repeat)
    dataset_yaml = {
        "path": str(args.output.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/dev" if frozen else "images/val",
        "nc": len(class_names),
        "names": {idx: name for idx, name in enumerate(class_names)},
        "channels": 3,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "steel_defect.yaml").write_text(
        yaml.safe_dump(dataset_yaml, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    split_manifest = {
        "source": str(args.source.resolve()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "calibration_ratio": args.calibration_ratio,
        "classes": class_names,
        "splits": {
            split: [str(record.image_path.relative_to(args.source)).replace("\\", "/") for record in items]
            for split, items in splits.items()
        },
        "groups": {
            split: sorted({record.group_id for record in items}) for split, items in splits.items()
        },
        "preparation": {
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "tile_layout": args.tile_layout,
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "calibration_ratio": args.calibration_ratio,
            "negative_ratio": args.negative_ratio,
            "rare_repeat_cap": args.rare_repeat_cap,
            "min_visibility": args.min_visibility,
            "deblur": args.deblur,
            "deblur_threshold": args.deblur_threshold,
            "clahe": args.clahe,
            "view_mode": args.view_mode,
            "global_size": args.global_size,
            "global_repeat": args.global_repeat,
            "partial_fragment_policy": "skip_views_with_any_unlabeled_intersection",
        },
    }
    (metadata_dir / "splits.json").write_text(
        json.dumps(({**frozen, "preparation": split_manifest["preparation"],
                     "frozen_source_manifest": str(Path(args.frozen_splits).resolve()),
                     "frozen_source_sha256": digest_file(Path(args.frozen_splits))}
                    if frozen else split_manifest), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (metadata_dir / "tiles.jsonl").open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "records": {split: len(items) for split, items in splits.items()},
        "tiles": dict(Counter(item["split"] for item in metadata)),
        "classes": class_names,
        "objects_before_repeat": {name: class_counts[name] for name in class_names},
        "original_objects_by_split": {
            split: {
                name: sum(
                    ann.class_id == class_id
                    for record in items
                    for ann in record.annotations
                )
                for class_id, name in enumerate(class_names)
            }
            for split, items in splits.items()
        },
        "repeated_train_tiles": sum(bool(item["is_repeat"]) for item in metadata),
        "global_views": sum(bool(item.get("is_global", False)) for item in metadata),
        "train_global_views": sum(item["split"] == "train" and bool(item.get("is_global", False)) for item in metadata),
        "train_crop_views": sum(item["split"] == "train" and not bool(item.get("is_global", False)) for item in metadata),
        "global_repeat": args.global_repeat,
        "view_mode": args.view_mode,
        "rare_repeat_cap": args.rare_repeat_cap,
        "deblur_enabled": args.deblur,
        "clahe_enabled": args.clahe,
    }
    (metadata_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
