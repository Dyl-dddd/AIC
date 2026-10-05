"""Build and audit the code-only optimized training update."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "payload"
ARCHIVE = HERE / "iron_4090_opt_v1_1_hotfix_code_only.zip"
SHA_FILE = HERE / "iron_4090_opt_v1_1_hotfix_code_only.zip.sha256"
REPORT = HERE / "BUILD_REPORT.json"
MANIFEST = PAYLOAD / "OPTIMIZED_UPDATE_MANIFEST.json"
ALLOWED = {
    "cloud_opt.py",
    "README_OPTIMIZED_4090.md",
    "configs/cloud/optimized_repeat3_4090.yaml",
    "configs/cloud/optimized_repeat3_aug_4090.yaml",
    "optimized/build_rare_repeat_manifest.py",
    "optimized/train.py",
}
FORBIDDEN_PARTS = {"data", "runs", "weights", ".venv", "__pycache__"}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    actual = {
        path.relative_to(PAYLOAD).as_posix()
        for path in PAYLOAD.rglob("*")
        if path.is_file() and path != MANIFEST and "__pycache__" not in path.parts
    }
    if actual != ALLOWED:
        raise ValueError(f"Unexpected payload files: missing={ALLOWED - actual}, extra={actual - ALLOWED}")

    files = []
    for relative in sorted(ALLOWED):
        path = PAYLOAD / relative
        files.append({"path": relative, "size": path.stat().st_size, "sha256": digest(path)})
    manifest = {
        "format": "iron-code-only-update-v1",
        "destination": "/mnt/proj/iron",
        "contains_dataset": False,
        "contains_weights": False,
        "files": files,
    }
    write_text(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    archive_files = sorted(actual | {MANIFEST.relative_to(PAYLOAD).as_posix()})
    for relative in archive_files:
        parts = set(PurePosixPath(relative).parts)
        if parts & FORBIDDEN_PARTS:
            raise ValueError(f"Forbidden content in code-only archive: {relative}")

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for relative in archive_files:
            bundle.write(PAYLOAD / relative, relative)

    with zipfile.ZipFile(ARCHIVE) as bundle:
        bad = bundle.testzip()
        names = bundle.namelist()
    if bad is not None or names != archive_files:
        raise ValueError(f"ZIP validation failed: bad={bad}, names={names}")

    archive_sha = digest(ARCHIVE)
    write_text(SHA_FILE, f"{archive_sha}  {ARCHIVE.name}\n")
    report = {
        "archive": str(ARCHIVE),
        "sha256": archive_sha,
        "size_bytes": ARCHIVE.stat().st_size,
        "entries": archive_files,
        "zip_test": "pass",
        "contains_dataset": False,
        "contains_weights": False,
    }
    write_text(REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
