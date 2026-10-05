"""Train one focused YOLO proposal expert and freeze-evaluate it, without Test access."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile

import torch
import ultralytics
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_retrain_pipeline import completed, run_logged, train_command
from scripts.run_semifinal_stage1 import SPLIT_SHA, verified_file, write_json
from steel_defect.classes import CLASS_NAMES
from steel_defect.runtime import sha256_file


MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"
ULTRALYTICS_VERSION = "8.3.169"
DATASET_NAME = "semifinal_yolo_targeted_v17"
RUN_NAME = "v17_targeted_rescue"
RESULT_NAME = "semifinal_v17_targeted_rescue_results_20261002.tar.gz"
CONFIG = ROOT / "configs/train/semifinal_v17_targeted_rescue_4090.yaml"


def focus_label(path: Path) -> bool:
    """Predeclared morphology targets; reads train labels only, never frozen dev labels."""
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        category = int(float(values[0]))
        width, height = float(values[3]), float(values[4])
        if category == 0 and width * height < 0.005:  # small jieba
            return True
        if category == 1 and height / max(width, 1e-8) > 4.0:  # slender zonglie
            return True
        if category == 3:  # jiaza class-confusion focus
            return True
    return False


def link_file(source: Path, destination: Path) -> None:
    """Cheap duplicate with no mutation of the source data."""
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        os.link(source, destination)


def prepare_focus_dataset(project: Path) -> dict:
    source = project / "data/semifinal_yolo_balanced_v3"
    destination = project / "data" / DATASET_NAME
    marker = destination / "metadata/summary.json"
    if marker.is_file():
        summary = json.loads(marker.read_text(encoding="utf-8"))
        if summary.get("source_dataset") != source.as_posix() or summary.get("schema") != 1:
            raise ValueError(f"existing target dataset has a different contract: {marker}")
        return summary
    if destination.exists():
        raise ValueError(f"partial target dataset exists; preserve and inspect it: {destination}")
    source_yaml = source / "steel_defect.yaml"
    if not source_yaml.is_file():
        raise FileNotFoundError(source_yaml)
    dataset = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if int(dataset["nc"]) != len(CLASS_NAMES):
        raise ValueError("source dataset has an unexpected class count")
    names = dataset["names"]
    names = [names[index] if index in names else names[str(index)] for index in range(len(CLASS_NAMES))] if isinstance(names, dict) else names
    if list(names) != CLASS_NAMES:
        raise ValueError(f"source class order mismatch: {names}")
    def image_and_label_folder(key: str) -> tuple[Path, Path]:
        relative = Path(str(dataset[key]))
        images = relative if relative.is_absolute() else source / relative
        if images.parent.name != "images":
            raise ValueError(f"unexpected {key} image path: {images}")
        labels = images.parent.parent / "labels" / images.name
        return images, labels

    src_train_images, src_train_labels = image_and_label_folder("train")
    src_dev_images, src_dev_labels = image_and_label_folder("val")
    for folder in (src_train_images, src_train_labels, src_dev_images, src_dev_labels):
        if not folder.is_dir():
            raise FileNotFoundError(folder)
    source_summary = source / "metadata/summary.json"
    if not source_summary.is_file():
        raise FileNotFoundError(source_summary)
    source_metadata = json.loads(source_summary.read_text(encoding="utf-8"))
    if int(source_metadata.get("global_repeat", -1)) != 3:
        raise ValueError("expected prepared balanced-v3 dataset with global_repeat=3")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise ValueError(f"staging directory already exists; inspect it: {staging}")
    for subfolder in ("images/train", "labels/train", "images/dev", "labels/dev", "metadata"):
        (staging / subfolder).mkdir(parents=True, exist_ok=False)
    counts = Counter()
    for image in sorted(src_train_images.iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = src_train_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        link_file(image, staging / "images/train" / image.name)
        link_file(label, staging / "labels/train" / label.name)
        counts["base_images"] += 1
        if focus_label(label):
            duplicate = f"{image.stem}__focus_v17"
            link_file(image, staging / "images/train" / f"{duplicate}{image.suffix}")
            link_file(label, staging / "labels/train" / f"{duplicate}.txt")
            counts["focus_duplicates"] += 1
    if counts["base_images"] < 100 or counts["focus_duplicates"] < 10:
        raise ValueError(f"unexpected training dataset counts: {counts}")
    for image in sorted(src_dev_images.iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = src_dev_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        link_file(image, staging / "images/dev" / image.name)
        link_file(label, staging / "labels/dev" / label.name)
        counts["dev_images"] += 1
    config = {
        "path": destination.as_posix(), "train": "images/train", "val": "images/dev",
        "nc": len(CLASS_NAMES), "names": {index: name for index, name in enumerate(CLASS_NAMES)},
        "channels": 3,
    }
    (staging / "steel_defect.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    summary = {
        "schema": 1, "source_dataset": source.as_posix(),
        "source_yaml_sha256": sha256_file(source_yaml),
        "source_summary_sha256": sha256_file(source_summary),
        **dict(counts), "focus_rule": "small jieba or slender zonglie or any jiaza; one extra copy",
        "test_access": False,
    }
    write_json(staging / "metadata/summary.json", summary)
    staging.rename(destination)
    return summary


def package_results(project: Path, output: Path, run_dir: Path, selection_dir: Path) -> dict:
    destination = project / RESULT_NAME
    if destination.exists():
        raise FileExistsError(f"result archive already exists: {destination}")
    files = [
        output / "preflight.json", output / "dataset_summary.json", output / "DECISION.json",
        selection_dir / "selection.json", selection_dir / "SUMMARY.md",
        run_dir / "training_contract.json", run_dir / "completion.json",
        run_dir / "initial_transfer.json", run_dir / "optimizer_steps.jsonl",
        run_dir / "batch_health.jsonl", run_dir / "args.yaml", run_dir / "results.csv",
        run_dir / "weights/last.pt",
    ]
    files.extend(path for path in (selection_dir / "cells").rglob("*") if path.is_file())
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing result files: {missing}")
    unique = list(dict.fromkeys(path.resolve() for path in files))
    manifest = {path.relative_to(project).as_posix(): sha256_file(path) for path in unique}
    manifest_path = output / "RESULT_MANIFEST.json"
    write_json(manifest_path, manifest)
    unique.append(manifest_path.resolve())
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"partial archive exists: {temporary}")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in unique:
            archive.add(path, arcname=path.relative_to(project).as_posix(), recursive=False)
    temporary.replace(destination)
    return {"archive": str(destination), "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination), "members": len(unique)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = project / "runs/semifinal/v17_targeted_rescue_pipeline"
    model_b = project / "runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt"
    split = project / "data/official_v2_20260831b/splits.json"
    source = project / "data/semifinal_yolo_balanced_v3/steel_defect.yaml"
    verified_file(model_b, MODEL_B_SHA256)
    verified_file(split, SPLIT_SHA)
    if not source.is_file() or not CONFIG.is_file():
        raise FileNotFoundError("missing balanced-v3 dataset or V17 training config")
    package_manifest = ROOT / "V17_PACKAGE_MANIFEST.json"
    if not package_manifest.is_file():
        raise FileNotFoundError(package_manifest)
    for relative, digest in json.loads(package_manifest.read_text(encoding="utf-8")).items():
        verified_file(ROOT / relative, digest)
    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(f"expected ultralytics {ULTRALYTICS_VERSION}, got {ultralytics.__version__}")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V17 requires the cloud RTX 4090")
    free, total = torch.cuda.mem_get_info()
    if free < 16 * 1024**3:
        raise RuntimeError(f"less than 16 GiB free VRAM: {free / 1024**3:.2f}")
    witness = torch.randn(64, 64, device="cuda", requires_grad=True)
    witness_loss = (witness @ witness.T).square().mean()
    witness_loss.backward()
    if not torch.isfinite(witness_loss) or not torch.isfinite(witness.grad).all():
        raise RuntimeError("CUDA witness failed")
    preflight = {
        "created": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "free_vram_gib": free / 1024**3,
        "total_vram_gib": total / 1024**3, "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "model_b_sha256": MODEL_B_SHA256,
        "split_sha256": SPLIT_SHA, "source_dataset": str(source),
        "config_sha256": sha256_file(CONFIG), "test_access": False,
        "package_manifest_sha256": sha256_file(package_manifest),
        "submission_creation": False,
    }
    del witness, witness_loss
    torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return
    dataset_summary = prepare_focus_dataset(project)
    write_json(output / "dataset_summary.json", dataset_summary)
    run_dir = project / "runs/semifinal" / RUN_NAME
    if not completed(run_dir):
        run_logged(train_command(sys.executable, project, CONFIG, run_dir, model_b),
                   project, output / "train.log")
    transfer = json.loads((run_dir / "initial_transfer.json").read_text(encoding="utf-8"))
    if transfer.get("source_sha256") != MODEL_B_SHA256:
        raise ValueError("training did not initialize from the verified Model B")
    if transfer.get("tensor_matches") != transfer.get("tensor_total"):
        raise ValueError("incomplete Model B transfer")
    last = run_dir / "weights/last.pt"
    if not last.is_file():
        raise FileNotFoundError(last)
    selection_dir = output / "selection"
    selection_file = selection_dir / "selection.json"
    if not selection_file.is_file():
        command = [
            sys.executable, "-u", str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
            "--project-root", str(project), "--run-dir", str(run_dir),
            "--output", str(selection_dir), "--imgsz", "1280", "--batch", "1",
            "--fixed-export-threshold", "0.0001", "--checkpoint", str(last),
            "--external-checkpoint", str(model_b),
        ]
        run_logged(command, project, output / "selection.log")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    decision = {
        "status": "RETURN_FOR_COMPLEMENTARITY_AUDIT",
        "reason": "Standalone dev score does not establish complementarity with V12 or official Test gain.",
        "selection": selection.get("selected"),
        "test_access": False, "submission_creation": False,
        "official_score_guaranteed": False,
    }
    write_json(output / "DECISION.json", decision)
    result = package_results(project, output, run_dir, selection_dir)
    print(json.dumps({"decision": decision, "result_archive": result}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
