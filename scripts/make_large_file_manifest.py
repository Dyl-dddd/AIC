"""Generate a manifest (relative path, size, SHA256) of large files for handoff.

Used before transferring data/weights via cloud drive, so the receiver can verify
integrity. Hashing multi-GB files takes time; use --min-size-mb and --ext to scope.

Example:
  python scripts/make_large_file_manifest.py \
      --dirs data delivery runs submissions fresh_20261004 analysis \
      --out LARGE_FILE_MANIFEST.tsv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_of(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-size-mb", type=float, default=0.0)
    parser.add_argument("--ext", nargs="*", default=None,
                        help="only include these extensions, e.g. .pt .json .zip")
    args = parser.parse_args()

    root = Path(args.root)
    min_bytes = args.min_size_mb * 1024 * 1024
    exts = {e.lower() for e in args.ext} if args.ext else None

    rows = []
    for d in args.dirs:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            size = p.stat().st_size
            if size < min_bytes:
                continue
            if exts and p.suffix.lower() not in exts:
                continue
            rel = p.relative_to(root).as_posix()
            print(f"hashing {rel} ({size/1024/1024:.1f} MB)")
            rows.append((rel, size, sha256_of(p)))

    rows.sort()
    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("path\tsize_bytes\tsha256\n")
        for rel, size, digest in rows:
            f.write(f"{rel}\t{size}\t{digest}\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
