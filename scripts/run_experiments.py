from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import DEFAULTS
from steel_defect.runtime import load_yaml
from steel_defect.training_audit import dataset_signature
from steel_defect.governance import digest_file


def completed_run(merged: dict) -> bool:
    config = {**DEFAULTS, **load_yaml(merged.get("config")), **{k: v for k, v in merged.items() if k != "config"}}
    root = Path(config["project"]) / config["name"]
    path = root / "completion.json"
    if not path.is_file() or not (root / "weights" / "best.pt").is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    code = {str(p): digest_file(p) for folder in ("steel_defect", "scripts") for p in sorted(Path(folder).glob("*.py"))}
    return (saved.get("status") == "complete" and saved.get("exit_code") == 0
            and saved.get("config") == config
            and saved.get("epochs_completed_this_invocation", 0) >= config["epochs"]
            and saved.get("dataset_signature") == dataset_signature(config["data"])
            and saved.get("code_sha256") == code)


def cli_args(values: dict) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            result.append(option if value else f"--no-{key.replace('_', '-')}")
        elif value is not None:
            result.extend((option, str(value)))
    return result


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable sequential ablation matrix.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun experiments that already have best.pt")
    args = parser.parse_args()

    payload = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    defaults = payload.get("defaults", {})
    experiments = payload.get("experiments", [])
    if not experiments:
        raise SystemExit("Experiment matrix is empty")
    state_path = args.matrix.with_suffix(".state.json")
    state = {
        "matrix": str(args.matrix),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": {},
    }
    if state_path.exists():
        state.update(json.loads(state_path.read_text(encoding="utf-8")))

    for experiment in experiments:
        merged = {**defaults, **experiment}
        name = str(merged["name"])
        best = Path(str(merged.get("project", "runs/steel"))) / name / "weights" / "best.pt"
        command = [sys.executable, "scripts/train.py", *cli_args(merged)]
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        if not args.force and completed_run(merged):
            state["experiments"][name] = {"status": "skipped_complete", "best": str(best)}
            write_state(state_path, state)
            continue
        print(" ".join(command), flush=True)
        state["experiments"][name] = {"status": "running", "command": command}
        write_state(state_path, state)
        completed = subprocess.run(command, check=False)
        status = "complete" if completed.returncode == 0 else "failed"
        state["experiments"][name] = {
            "status": status,
            "returncode": completed.returncode,
            "command": command,
            "best": str(best) if best.is_file() else None,
        }
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_state(state_path, state)
        if completed.returncode and not args.continue_on_error:
            raise SystemExit(completed.returncode)
    if not args.dry_run:
        write_state(state_path, state)
        print(f"Experiment state: {state_path.resolve()}")


if __name__ == "__main__":
    main()
