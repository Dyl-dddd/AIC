"""Read-only checks before the code-only semifinal retraining package runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


EXPECTED_WEIGHT_SHA256 = "5852470665ee87c29ac741e6dd6b786ec51c10e6f7ebf79d05d9e70ecf7562c7"
EXPECTED_CLASSES = (
    "jieba", "zonglie", "qilie", "jiaza", "yiwuyaru", "huashang",
    "mamianmakeng", "yanghuatiepi", "gunyin",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def normalized_names(payload: dict) -> tuple[str, ...]:
    names = payload.get("names")
    if isinstance(names, dict):
        return tuple(str(names[key]) for key in sorted(names, key=int))
    if isinstance(names, list):
        return tuple(str(item) for item in names)
    raise ValueError("Dataset YAML must contain ordered class names")


def inspect(config_path: Path, weight_override: Path | None, skip_cuda: bool) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["train"]
    dataset_path = Path(config["data"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Prepared dataset YAML missing: {dataset_path}; run scripts/prepare_data.py first")
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    names = normalized_names(dataset)
    if names != EXPECTED_CLASSES or int(dataset["nc"]) != len(EXPECTED_CLASSES):
        raise ValueError(f"Training class order differs from frozen 9-class checkpoint: {names}")
    root = Path(dataset["path"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset path inside {dataset_path} does not exist: {root}. "
            "Regenerate the prepared data on this machine; do not copy a Windows absolute path into Linux."
        )
    if any(part.lower() == "test" for part in root.parts):
        raise ValueError(f"Refusing a test-set data root: {root}")
    split_files: dict[str, set[str]] = {}
    for split in ("train", "val"):
        relative = dataset.get(split)
        if not isinstance(relative, str):
            raise ValueError(f"Dataset YAML lacks {split} directory")
        folder = root / relative
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing {split} images directory: {folder}")
        split_files[split] = {path.name for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES}
        if not split_files[split]:
            raise ValueError(f"No {split} images found under {folder}")
    repeated = split_files["train"] & split_files["val"]
    if repeated:
        raise ValueError(f"Train/val image filename overlap: {len(repeated)}; e.g. {next(iter(repeated))}")
    weight = weight_override or Path(config["model"])
    if not weight.is_file():
        raise FileNotFoundError(f"Initialization checkpoint missing: {weight}")
    actual_hash = digest(weight)
    if actual_hash != EXPECTED_WEIGHT_SHA256:
        raise ValueError(f"Checkpoint SHA256 differs from verified epoch25: {actual_hash}")
    cuda_info = None
    if not skip_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot see a CUDA GPU; do not start a long training run")
        device = torch.device("cuda:0")
        cuda_info = {
            "name": torch.cuda.get_device_name(device),
            "vram_gib": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2),
        }
        (torch.ones(1, device=device) + 1).cpu()
    return {
        "ready": True,
        "config": str(config_path),
        "dataset": str(dataset_path),
        "train_images": len(split_files["train"]),
        "val_images": len(split_files["val"]),
        "checkpoint": str(weight),
        "checkpoint_sha256": actual_hash,
        "cuda": cuda_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/semifinal_retrain_v2_4090.yaml"))
    parser.add_argument("--weight", type=Path, help="Override the model path only for preflight")
    parser.add_argument("--skip-cuda", action="store_true", help="For code-only verification on a CPU host")
    args = parser.parse_args()
    print(json.dumps(inspect(args.config, args.weight, args.skip_cuda), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
