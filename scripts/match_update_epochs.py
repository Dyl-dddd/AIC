"""Estimate optimizer calls using installed Ultralytics 8.3 accumulation semantics.

AMP can skip steps: actual optimizer_steps.jsonl is authoritative, not this estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np


def estimate_updates(samples, batch, epochs, nbs=64, warmup_epochs=3.0):
    if min(samples, batch, epochs, nbs) <= 0 or warmup_epochs < 0:
        raise ValueError("Counts must be positive; warmup non-negative")
    batches = math.ceil(samples / batch)
    warmup = max(round(warmup_epochs * batches), 100) if warmup_epochs > 0 else -1
    last, updates = -1, 0
    accumulate = max(round(nbs / batch), 1)
    for index in range(batches * epochs):
        if index <= warmup:
            accumulate = max(1, int(np.interp(index, [0, warmup], [1, nbs / batch]).round()))
        if index - last >= accumulate:
            updates += 1
            last = index
    return {"microbatches": batches * epochs, "estimated_optimizer_calls": updates}


def train_samples(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload["tiles"]["train"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-epochs", type=int, required=True)
    parser.add_argument("--reference-batch", type=int, required=True)
    parser.add_argument("--candidate-batch", type=int, required=True)
    parser.add_argument("--reference-nbs", type=int, default=64)
    parser.add_argument("--candidate-nbs", type=int, default=64)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    args = parser.parse_args()
    reference = estimate_updates(train_samples(args.reference_summary), args.reference_batch,
                                 args.reference_epochs, args.reference_nbs, args.warmup_epochs)
    samples = train_samples(args.candidate_summary)
    low, high = 1, 1
    def candidate(epochs):
        return estimate_updates(samples, args.candidate_batch, epochs, args.candidate_nbs, args.warmup_epochs)
    target = reference["estimated_optimizer_calls"]
    while candidate(high)["estimated_optimizer_calls"] < target:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if candidate(mid)["estimated_optimizer_calls"] < target:
            low = mid + 1
        else:
            high = mid
    choices = {max(1, low - 1), low}
    epochs = min(choices, key=lambda e: abs(candidate(e)["estimated_optimizer_calls"] - target))
    print(json.dumps({"reference": reference, "candidate": candidate(epochs), "candidate_epochs": epochs,
                      "warning": "Estimated calls, not actual successful AMP optimizer updates"}, indent=2))


if __name__ == "__main__":
    main()
