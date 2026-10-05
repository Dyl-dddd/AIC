"""Build auditable single-model interpolations of two related YOLO checkpoints."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import torch

from steel_defect.runtime import sha256_file


def interpolate_state_dicts(old: dict, new: dict, alpha: float) -> dict:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if old.keys() != new.keys():
        raise ValueError("checkpoint state-dict keys differ")
    result = {}
    for key in old:
        left, right = old[key], new[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"incompatible tensor: {key}")
        if torch.is_floating_point(left):
            result[key] = left.float().mul(1.0 - alpha).add(right.float(), alpha=alpha).to(left.dtype)
        else:
            if not torch.equal(left, right):
                result[key] = right.clone() if alpha >= 0.5 else left.clone()
            else:
                result[key] = left.clone()
    return result


def _model(checkpoint: dict):
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError("checkpoint has neither ema nor model")
    return model


def build_soup(old_path: Path, new_path: Path, alpha: float, output: Path) -> dict:
    old_path, new_path, output = old_path.resolve(), new_path.resolve(), output.resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    expected = {
        "schema": 1,
        "alpha_new": alpha,
        "old_sha256": sha256_file(old_path),
        "new_sha256": sha256_file(new_path),
    }
    if output.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected.items()):
            if existing.get("output_sha256") == sha256_file(output):
                return existing
        raise ValueError(f"existing soup does not match requested parents/alpha: {output}")

    old_checkpoint = torch.load(old_path, map_location="cpu", weights_only=False)
    new_checkpoint = torch.load(new_path, map_location="cpu", weights_only=False)
    old_model = _model(old_checkpoint).float()
    new_model = _model(new_checkpoint).float()
    old_state, new_state = old_model.state_dict(), new_model.state_dict()
    interpolated = interpolate_state_dicts(old_state, new_state, alpha)
    soup_model = copy.deepcopy(old_model)
    soup_model.load_state_dict(interpolated, strict=True)
    soup_model.half()

    checkpoint = dict(old_checkpoint)
    checkpoint.update({
        "epoch": -1,
        "best_fitness": None,
        "model": None,
        "ema": soup_model,
        "updates": 0,
        "optimizer": None,
        "train_results": None,
        "date": datetime.now(timezone.utc).isoformat(),
        "soup_metadata": expected,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(checkpoint, temporary)
        roundtrip = torch.load(temporary, map_location="cpu", weights_only=False)
        state = _model(roundtrip).state_dict()
        if state.keys() != interpolated.keys() or any(state[key].shape != interpolated[key].shape for key in state):
            raise ValueError("round-trip soup validation failed")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {**expected, "output": str(output), "output_sha256": sha256_file(output),
                "parameter_tensors": len(interpolated),
                "parameters": sum(value.numel() for value in interpolated.values())}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata

