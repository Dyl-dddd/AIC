from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal_grounding.data import GroundingDataset, check_files, collate_grounding, load_annotations
from multimodal_grounding.metrics import sanitize_bbox
from multimodal_grounding.model import TextGuidedBoxRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict normalized bbox fields and write official JSON.")
    parser.add_argument("--data", required=True, help="Path to official JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint from scripts/mm_train.py.")
    parser.add_argument("--root", default=None, help="Dataset root. Defaults to the JSON parent.")
    parser.add_argument("--output", default="submission.json", help="Output JSON path.")
    parser.add_argument("--zip", dest="zip_path", default=None, help="Optional zip file path for platform upload.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=None, help="Defaults to checkpoint --img-size.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    train_args = checkpoint.get("args", {})
    image_size = args.img_size or int(train_args.get("img_size", 384))
    max_tokens = int(train_args.get("max_tokens", 32))
    vocab_size = int(model_config.get("vocab_size", train_args.get("vocab_size", 8192)))

    samples = load_annotations(args.data, root=args.root, require_bbox=False)
    missing = check_files(samples)
    if missing:
        raise FileNotFoundError(f"Missing modality file examples: {missing[:5]}")

    dataset = GroundingDataset(
        samples=samples,
        image_size=image_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        augment=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_grounding,
    )

    model = TextGuidedBoxRegressor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    predictions: dict[str, dict] = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            pred = model(images, tokens, mask).cpu().tolist()
            for sample, bbox in zip(batch["samples"], pred, strict=True):
                predictions[sample.sample_id] = sample.prediction_record(sanitize_bbox(bbox))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(predictions)} predictions to {output}")

    if args.zip_path:
        zip_path = Path(args.zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output, arcname=output.name)
        print(f"wrote zip package to {zip_path}")


if __name__ == "__main__":
    main()
