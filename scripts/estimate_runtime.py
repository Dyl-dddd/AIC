"""Estimate full training duration from an Ultralytics pilot run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate runtime from results.csv after a 3-5 epoch pilot")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--safety", type=float, default=1.15)
    args = parser.parse_args()
    with args.results.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("results.csv has no completed epochs")
    completed = int(float(rows[-1]["epoch"]))
    elapsed_seconds = float(rows[-1]["time"])
    hours = elapsed_seconds / max(completed, 1) * args.total_epochs * args.safety / 3600
    print(
        f"pilot={completed} epochs, elapsed={elapsed_seconds / 60:.2f} min, "
        f"estimated {args.total_epochs} epochs={hours:.2f} h (safety x{args.safety:g})"
    )


if __name__ == "__main__":
    main()
