"""Local RTX 4060 adapter for the audited cloud experiment controller."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
CLOUD_SOURCE = WORKSPACE / "delivery" / "iron_4090_20260831" / "iron" / "cloud.py"
PROFILE = WORKSPACE / "configs" / "cloud" / "local4060.yaml"
SPLIT = Path("data/official_v2_20260831b/splits.json")
GATE = Path("runs/diagnostic_v2/R002_stock_bce_fp32_lr1e4/learning_gate.json")
DATA_YAMLS = (
    Path("data/official_yolo_hybrid_v2/steel_defect.yaml"),
    Path("data/diagnostic_v2/diagnostic.yaml"),
)


def _load_controller():
    spec = importlib.util.spec_from_file_location("iron_cloud_controller", CLOUD_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load experiment controller: {CLOUD_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_local(full: bool = False) -> None:
    required = [PROFILE, WORKSPACE / SPLIT, WORKSPACE / GATE, WORKSPACE / "yolo11s.pt"]
    required.extend(WORKSPACE / path for path in DATA_YAMLS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required local files:\n" + "\n".join(missing))

    split_spec = json.loads((WORKSPACE / SPLIT).read_text(encoding="utf-8"))
    for yaml_path in DATA_YAMLS:
        spec = yaml.safe_load((WORKSPACE / yaml_path).read_text(encoding="utf-8"))
        root = Path(spec["path"])
        if not root.is_absolute():
            root = (WORKSPACE / yaml_path).parent / root
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        for key in ("train", "val"):
            values = spec.get(key, [])
            for value in ([values] if isinstance(values, str) else values):
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = root / candidate
                if not candidate.exists():
                    raise FileNotFoundError(f"Dataset split not found: {candidate}")

    gate = json.loads((WORKSPACE / GATE).read_text(encoding="utf-8"))
    if not gate.get("gate", {}).get("passed"):
        raise RuntimeError("Diagnostic learning gate has not passed")
    splits = split_spec.get("splits", {})
    print(
        "PASS local data: "
        f"train={len(splits.get('train', []))}, "
        f"dev={len(splits.get('dev', []))}, "
        f"final={len(splits.get('final', []))}; "
        f"full_hash={full}",
        flush=True,
    )


def preflight_local() -> None:
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Python 3.11 or 3.12 required")

    import cv2
    import torch
    import torchvision
    from steel_defect.training_audit import cuda_witness

    expected = {
        "torch": "2.5.1",
        "torchvision": "0.20.1",
        "ultralytics": "8.3.169",
        "numpy": "1.26.4",
        "opencv-python": "4.11.0.86",
        "Pillow": "10.4.0",
        "PyYAML": "6.0.1",
    }
    installed = {name: importlib.metadata.version(name) for name in expected}
    mismatches = {
        name: {"expected": version, "installed": installed[name]}
        for name, version in expected.items()
        if installed[name].split("+")[0] != version
    }
    if mismatches:
        raise RuntimeError(f"Environment differs from tested core: {mismatches}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU training")

    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    if total < 7 * 1024**3:
        raise RuntimeError("Local profile requires at least 7 GiB visible VRAM")
    if free < 5 * 1024**3:
        raise RuntimeError(
            f"Insufficient free VRAM: {free / 1024**3:.2f} GiB; close GPU-heavy applications"
        )
    disk_free = shutil.disk_usage(WORKSPACE).free
    if disk_free < 10 * 1024**3:
        raise RuntimeError("Keep at least 10 GiB free for checkpoints and logs")

    witness = cuda_witness(42)
    keep = torchvision.ops.nms(
        torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda"),
        torch.tensor([0.9], device="cuda"),
        0.5,
    )
    if keep.cpu().tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS witness failed")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "versions": installed,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "vram_gib": total / 1024**3,
        "free_vram_gib": free / 1024**3,
        "disk_free_gib": disk_free / 1024**3,
        "cuda_witness": witness,
        "profile_sha256": _sha256(PROFILE),
    }
    report_path = WORKSPACE / ".aris" / "compute" / "preflight_local.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def preflight_subprocess() -> None:
    """Run CUDA checks out of process so the controller does not retain PyTorch RAM."""
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_local-preflight"],
        cwd=WORKSPACE,
        check=True,
    )


def assert_training_complete_local(root: Path, count: int) -> None:
    """Accept missing deferred-validation cells while strictly checking training loss."""
    import csv
    import math

    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "complete":
        raise ValueError("Training did not finish successfully")
    with (root / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != count or [int(float(row["epoch"])) for row in rows] != list(range(1, count + 1)):
        raise ValueError("Epoch history is incomplete")
    for row in rows:
        for key, value in row.items():
            if key.startswith("train/") and (value in (None, "") or not math.isfinite(float(value))):
                raise ValueError(f"Non-finite training history: epoch={row['epoch']} field={key}")
            if key.startswith("val/") and value not in (None, "") and not math.isfinite(float(value)):
                raise ValueError(f"Non-finite validation history: epoch={row['epoch']} field={key}")


def main() -> None:
    os.chdir(WORKSPACE)
    if len(sys.argv) == 2 and sys.argv[1] == "_local-preflight":
        preflight_local()
        return
    controller = _load_controller()
    controller.ROOT = WORKSPACE
    controller.PROFILE = PROFILE
    controller.SPLIT = SPLIT
    controller.GATE = GATE
    controller.verify = verify_local
    controller.preflight = preflight_subprocess
    controller.assert_training_complete = assert_training_complete_local
    controller.main()


if __name__ == "__main__":
    main()
