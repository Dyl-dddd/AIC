from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal_grounding.data import check_files, load_annotations
from multimodal_grounding.model import TextGuidedBoxRegressor
from multimodal_grounding.train_utils import (
    evaluate,
    grounding_loss,
    make_loader,
    save_checkpoint,
    set_seed,
    split_by_visible_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a text-guided multimodal bbox regressor.")
    parser.add_argument("--data", required=True, help="Path to official train JSON.")
    parser.add_argument("--root", default=None, help="Dataset root. Defaults to the JSON parent.")
    parser.add_argument("--output", default="runs/multimodal_grounding/baseline", help="Output run directory.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--text-dim", type=int, default=128)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--giou-weight", type=float, default=1.0)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    samples = load_annotations(args.data, root=args.root, require_bbox=True)
    missing = check_files(samples)
    if missing:
        raise FileNotFoundError(f"Missing modality file examples: {missing[:5]}")

    train_samples, val_samples = split_by_visible_path(samples, args.val_ratio, args.seed)
    if train_samples == val_samples:
        print("warning: using the same samples for train and validation because the dataset is too small.")

    write_json(
        output / "run_config.json",
        {
            **vars(args),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        },
    )

    train_loader = make_loader(
        train_samples,
        image_size=args.img_size,
        max_tokens=args.max_tokens,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        workers=args.workers,
        augment=not args.no_augment,
        shuffle=True,
    )
    val_loader = make_loader(
        val_samples,
        image_size=args.img_size,
        max_tokens=args.max_tokens,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        workers=args.workers,
        augment=False,
        shuffle=False,
    )

    device = choose_device(args.device)
    model = TextGuidedBoxRegressor(
        input_channels=7,
        vocab_size=args.vocab_size,
        text_dim=args.text_dim,
        width=args.width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in train_loader:
            target = batch["bbox"]
            if target is None:
                continue
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(images, tokens, mask)
            loss = grounding_loss(pred, target, args.l1_weight, args.giou_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = images.shape[0]
            train_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = train_loss / max(seen, 1)
        metrics = evaluate(model, val_loader, device, args.l1_weight, args.giou_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_iou": metrics["iou"],
            "val_acc05": metrics["acc05"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_iou={metrics['iou']:.4f} "
            f"val_acc05={metrics['acc05']:.4f}"
        )

        save_checkpoint(output / "last.pt", model, optimizer, epoch, metrics, vars(args))
        is_best = metrics["acc05"] > best_acc or (metrics["acc05"] == best_acc and metrics["loss"] < best_loss)
        if is_best:
            best_acc = metrics["acc05"]
            best_loss = metrics["loss"]
            save_checkpoint(output / "best.pt", model, optimizer, epoch, metrics, vars(args))

    write_json(output / "history.json", history)
    print(f"saved: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
