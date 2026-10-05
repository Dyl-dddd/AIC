"""Generate submit-only V15 class attribution probes from V12 and V14 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile


TARGET_CLASSES = ("jieba", "jiaza", "mamianmakeng")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_package(root: Path) -> str:
    path = root / "V15_PACKAGE_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        member = root / relative
        if not member.is_file() or sha256_file(member) != expected:
            raise ValueError(f"package member mismatch: {relative}")
    return sha256_file(path)


def load_submission(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"submission must be a non-empty JSON list: {path}")
    required = {"image_id", "category_name", "bbox", "score"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"invalid row {index} in {path}")
        score = float(row["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score at row {index} in {path}: {score}")
    return rows


def identity_key(row: dict) -> tuple[str, str, tuple[float, ...]]:
    bbox = row["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"invalid bbox: {bbox!r}")
    return str(row["image_id"]), str(row["category_name"]), tuple(float(value) for value in bbox)


def identity_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(identity_key(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_class_probe(v12: list[dict], v14: list[dict], target: str) -> list[dict]:
    output: list[dict] = []
    for old, new in zip(v12, v14, strict=True):
        item = dict(old)
        if str(old["category_name"]) == target:
            item["score"] = float(new["score"])
        output.append(item)
    return output


def build_blend(v12: list[dict], v14: list[dict], alpha: float) -> list[dict]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    output: list[dict] = []
    for old, new in zip(v12, v14, strict=True):
        item = dict(old)
        if str(old["category_name"]) in TARGET_CLASSES:
            old_score = max(float(old["score"]), 1e-12)
            new_score = max(float(new["score"]), 1e-12)
            item["score"] = old_score ** (1.0 - alpha) * new_score**alpha
        output.append(item)
    return output


def write_submission(rows: list[dict], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def package_submission(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(source, "submission.json")
    with zipfile.ZipFile(temporary) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"invalid submit-only ZIP: {destination}")
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--v12", type=Path)
    parser.add_argument("--v14", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    package_manifest_sha256 = verify_package(root)
    project = args.project_root.resolve()
    v12_path = args.v12 or project / "runs/semifinal/v12_perclass_ranker_submission/submission.json"
    v14_path = args.v14 or project / "runs/semifinal/v14_v13_hybrid_ranker_submission/submission.json"
    v12 = load_submission(v12_path)
    v14 = load_submission(v14_path)
    if len(v12) != len(v14):
        raise ValueError(f"row count changed: V12={len(v12)}, V14={len(v14)}")
    identity = identity_digest(v12)
    if identity_digest(v14) != identity:
        raise ValueError("V12 and V14 membership/order/geometry differ; refusing to build probes")

    changed_classes = sorted({
        str(old["category_name"])
        for old, new in zip(v12, v14, strict=True)
        if float(old["score"]) != float(new["score"])
    })
    if changed_classes != sorted(TARGET_CLASSES):
        raise ValueError(f"unexpected changed classes: {changed_classes}")

    variants: dict[str, list[dict]] = {
        f"probe_{class_name}": build_class_probe(v12, v14, class_name)
        for class_name in TARGET_CLASSES
    }
    variants["blend25_all3"] = build_blend(v12, v14, 0.25)
    variants["blend50_all3"] = build_blend(v12, v14, 0.50)

    output_root = project / "runs/semifinal/v15_class_attribution_submission"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package_manifest_sha256": package_manifest_sha256,
        "v12": {"path": str(v12_path), "sha256": sha256_file(v12_path)},
        "v14": {"path": str(v14_path), "sha256": sha256_file(v14_path)},
        "row_count": len(v12),
        "identity_sha256": identity,
        "changed_classes": changed_classes,
        "membership_and_geometry_frozen": True,
        "variants": {},
    }
    for name, rows in variants.items():
        if identity_digest(rows) != identity:
            raise RuntimeError(f"identity changed in {name}")
        variant_root = output_root / name
        variant_root.mkdir(parents=True, exist_ok=True)
        submission = variant_root / "submission.json"
        archive = output_root / f"semifinal_v15_{name}_SUBMIT_ONLY.zip"
        write_submission(rows, submission)
        package_submission(submission, archive)
        changed_rows = sum(
            float(old["score"]) != float(new["score"])
            for old, new in zip(v12, rows, strict=True)
        )
        manifest["variants"][name] = {
            "changed_rows_vs_v12": changed_rows,
            "submission_sha256": sha256_file(submission),
            "zip": str(archive),
            "zip_sha256": sha256_file(archive),
        }
        print(f"READY {name}: {archive}", flush=True)

    audit = output_root / "V15_OUTPUT_MANIFEST.json"
    write_submission(manifest, audit) if isinstance(manifest, list) else audit.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
