"""Experiment configuration and reproducibility helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import torch
import ultralytics
import yaml


def load_yaml_section(path: str | Path | None, section: str | None = None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    if section is not None:
        payload = payload.get(section, payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration section {section!r} must be a mapping: {path}")
    return payload


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    """Backward-compatible training-profile loader."""
    return load_yaml_section(path, "train")


def apply_profile_defaults(parser, section: str):
    """Load a YAML section as argparse defaults while preserving CLI precedence."""
    preview, _ = parser.parse_known_args()
    config_path = getattr(preview, "config", None)
    if config_path is not None:
        profile = load_yaml_section(config_path, section)
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(profile) - valid)
        if unknown:
            raise ValueError(f"Unknown {section} profile keys: {unknown}")
        parser.set_defaults(**profile)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def platform_manifest(config: dict[str, Any]) -> dict[str, Any]:
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "index": index,
                    "name": properties.name,
                    "memory_gib": round(properties.total_memory / 1024**3, 3),
                    "capability": [properties.major, properties.minor],
                }
            )
    paths = [config.get("data"), config.get("model")]
    pretrained = config.get("pretrained")
    if pretrained and str(pretrained).lower() != "none":
        paths.append(pretrained)
    hashes = {str(path): sha256_file(path) for path in paths if path}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "artifact_sha256": hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpu,
        "distributed_environment": {
            key: os.environ[key]
            for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "SLURM_JOB_ID")
            if key in os.environ
        },
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
