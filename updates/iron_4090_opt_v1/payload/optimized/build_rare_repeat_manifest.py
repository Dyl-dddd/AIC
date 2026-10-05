"""Create a rare-class repeated training list without copying image data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import yaml


CLASS_NAMES = [
    "jieba", "zonglie", "qilie", "jiaza", "yiwuyaru",
    "huashang", "mamianmakeng", "yanghuatiepi", "gunyin",
]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repeat-cap", type=float, default=3.0)
    parser.add_argument("--output-list", type=Path, required=True)
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1.0 < args.repeat_cap <= 3.0:
        parser.error("--repeat-cap must be within (1, 3]")

    dataset_root = args.dataset_root.resolve()
    metadata_path = dataset_root / "metadata" / "tiles.jsonl"
    summary_path = dataset_root / "metadata" / "summary.json"
    if not metadata_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Existing hybrid-v2 metadata is missing; do not regenerate source data")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("records") != {"train": 2395, "dev": 459}:
        raise ValueError(f"Unexpected frozen data summary: {summary.get('records')}")

    items = [json.loads(line) for line in metadata_path.open(encoding="utf-8")]
    train_items = [item for item in items if item["split"] == "train"]
    if len(train_items) != int(summary["tiles"]["train"]):
        raise ValueError("Training-view count disagrees with frozen summary")
    presence: Counter[int] = Counter()
    for item in train_items:
        if not item.get("is_global", False):
            presence.update(item["class_ids"])
    majority = max(presence.values())

    lines = []
    repeated_by_class: Counter[int] = Counter()
    repeated_views = 0
    missing = []
    for item in train_items:
        image = dataset_root / "images" / "train" / f"{item['tile']}.jpg"
        label = dataset_root / "labels" / "train" / f"{item['tile']}.txt"
        if not image.is_file() or not label.is_file():
            missing.append(str(image if not image.is_file() else label))
            continue
        copies = 1
        if item["class_ids"] and not item.get("is_global", False):
            factor = max(math.sqrt(majority / presence[class_id]) for class_id in item["class_ids"])
            copies += max(0, int(round(min(args.repeat_cap, factor))) - 1)
        lines.extend([image.as_posix()] * copies)
        if copies > 1:
            repeated_views += copies - 1
            for class_id in item["class_ids"]:
                repeated_by_class[class_id] += copies - 1
    if missing:
        raise FileNotFoundError("Missing prepared views:\n" + "\n".join(missing[:20]))

    atomic_text(args.output_list, "\n".join(lines) + "\n")
    yaml_spec = {
        "path": dataset_root.as_posix(),
        "train": args.output_list.resolve().as_posix(),
        "val": "images/dev",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "channels": 3,
    }
    atomic_text(args.output_yaml, yaml.safe_dump(yaml_spec, sort_keys=False, allow_unicode=True))
    report = {
        "dataset_root": dataset_root.as_posix(),
        "base_training_views": len(train_items),
        "repeated_views": repeated_views,
        "effective_training_views": len(lines),
        "growth_fraction": repeated_views / len(train_items),
        "repeat_cap": args.repeat_cap,
        "class_tile_presence": {CLASS_NAMES[k]: v for k, v in sorted(presence.items())},
        "extra_presence_by_class": {CLASS_NAMES[k]: v for k, v in sorted(repeated_by_class.items())},
        "list": args.output_list.resolve().as_posix(),
        "yaml": args.output_yaml.resolve().as_posix(),
    }
    atomic_text(args.report, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
