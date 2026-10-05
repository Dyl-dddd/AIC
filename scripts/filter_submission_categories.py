"""Remove unsupported categories from a large JSON-array submission.

This intentionally drops predictions rather than relabeling them as another defect.
The output is a new file; the input is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from validate_large_submission import iter_json_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-category", action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if source == output:
        raise ValueError("Output must differ from source")
    if output.exists() or args.report.exists():
        raise FileExistsError("Output or report already exists; refusing to overwrite")

    excluded = set(args.exclude_category)
    counts: Counter[str] = Counter()
    kept = 0
    removed = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write("[")
        for item in iter_json_array(source):
            category = item["category_name"]
            counts[category] += 1
            if category in excluded:
                removed += 1
                continue
            if kept:
                handle.write(",")
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            kept += 1
        handle.write("]\n")

    report = {
        "source": str(source),
        "output": str(output),
        "excluded_categories": sorted(excluded),
        "source_predictions": sum(counts.values()),
        "kept_predictions": kept,
        "removed_predictions": removed,
        "source_class_counts": dict(counts),
        "output_bytes": output.stat().st_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
