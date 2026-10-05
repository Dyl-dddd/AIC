"""Remove a preregistered low-score tail from a validated single-JSON submission ZIP."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def read_submission(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise ValueError(f"source must contain exactly one root submission.json: {path}")
        rows = json.loads(archive.read("submission.json"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("submission must be a non-empty JSON list")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quantile", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("quantile must be in (0,1)")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_submission(args.source)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("source scores are not finite")
    floor = float(np.quantile(scores, args.quantile))
    kept, removed = [], Counter()
    for row in rows:
        if float(row["score"]) >= floor:
            kept.append(row)
        else:
            removed[row["category_name"]] += 1
    if not kept or len(kept) == len(rows):
        raise ValueError("filter had no valid effect")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(kept, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("submission.json", encoded)
    with zipfile.ZipFile(args.output) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("output ZIP failed integrity check")
    print(json.dumps({
        "output": str(args.output), "output_sha256": sha256(args.output),
        "source": str(args.source), "source_sha256": sha256(args.source),
        "quantile": args.quantile, "score_floor": floor,
        "before": len(rows), "after": len(kept), "removed": len(rows) - len(kept),
        "removed_by_class": dict(removed), "single_root_json": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
