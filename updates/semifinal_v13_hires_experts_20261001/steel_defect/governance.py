"""Local-only dataset provenance and conservative connected-component grouping."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import base64
import numpy as np


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_keys(name: str) -> list[str]:
    stem = Path(name).stem
    raw = re.split(r"[-_]Raw\d+", stem, maxsplit=1, flags=re.I)[0]
    keys = ["raw:" + raw] if raw != stem else []
    # C is a conservative candidate relation, NOT a verified physical coil ID.
    candidate = re.match(r"^(C[^_]+)_V[^_]+_F", stem, re.I)
    if candidate:
        keys.append("candidate_c:" + candidate.group(1).casefold())
    return keys


def connected_groups(entries: list[dict], near_distance: int = 4) -> tuple[dict[str, str], list[dict]]:
    """Union source keys, byte/pixel equality and verified close dHashes.

    Perceptual matches are only isolation relations, never label-merging evidence.
    Include test nodes so transitive test-connected training groups are excluded.
    """
    parents = {entry["id"]: entry["id"] for entry in entries}

    def root(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)

    seen = {}
    for entry in entries:
        # Test is linked through content evidence only. A shared filename prefix
        # with test is reported separately: it does not establish image identity.
        keys = source_keys(entry["path"]) if entry.get("kind") != "test" else []
        keys += [prefix + str(entry[field]) for prefix, field in
                 (("bytes:", "sha256"), ("pixels:", "pixel_sha256")) if entry.get(field)]
        for key in keys:
            if key in seen:
                union(entry["id"], seen[key])
            else:
                seen[key] = entry["id"]
    near = []
    if near_distance >= 0:
        for index, a in enumerate(entries):
            if "dhash" not in a:
                continue
            for b in entries[:index]:
                if "dhash" not in b or (a["width"], a["height"]) != (b["width"], b["height"]):
                    continue
                if a.get("pixel_sha256") == b.get("pixel_sha256"):
                    continue
                distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
                if distance <= near_distance:
                    # Uniform/repetitive steel textures make dHash alone unsafe.
                    verified = False
                    if a.get("thumbnail") and b.get("thumbnail"):
                        ta = np.frombuffer(base64.b64decode(a["thumbnail"]), np.uint8).astype(float)
                        tb = np.frombuffer(base64.b64decode(b["thumbnail"]), np.uint8).astype(float)
                        if min(ta.std(), tb.std()) >= 4:
                            correlation = float(np.corrcoef(ta, tb)[0, 1])
                            rmse = float(np.sqrt(np.mean((ta - tb) ** 2)))
                            verified = correlation >= .9995 and rmse <= 1.0
                    if verified:
                        near.append({"a": a["id"], "b": b["id"], "hamming": distance,
                                     "correlation": correlation, "thumbnail_rmse": rmse})
                        union(a["id"], b["id"])
    members = defaultdict(list)
    for key in parents:
        members[root(key)].append(key)
    group_ids = {key: hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()[:20]
                 for key, items in members.items()}
    return {key: group_ids[root(key)] for key in parents}, near


def validate_split_manifest(manifest: dict) -> None:
    seen_paths, seen_groups = set(), set()
    for split, paths in manifest["splits"].items():
        if not paths or len(paths) != len(set(paths)):
            raise ValueError(f"Empty or duplicate paths in split {split}")
        groups = set(manifest["groups"][split])
        if seen_paths.intersection(paths) or seen_groups.intersection(groups):
            raise ValueError("Cross-split path/group leakage")
        seen_paths.update(paths)
        seen_groups.update(groups)


def frozen_records(records, source: Path, manifest: dict, verify: bool = True):
    from dataclasses import replace
    validate_split_manifest(manifest)
    lookup = {r.image_path.relative_to(source).as_posix(): r for r in records}
    result = {}
    for split, paths in manifest["splits"].items():
        result[split] = []
        for path in paths:
            if path not in lookup:
                raise ValueError(f"Frozen split missing source: {path}")
            item = manifest["records"][path]
            record = lookup[path]
            if item["group"] not in manifest["groups"][split]:
                raise ValueError(f"Group mismatch for {path}")
            if verify and (digest_file(record.image_path) != item["sha256"] or
                           digest_file(record.xml_path) != item["xml_sha256"]):
                raise ValueError(f"Frozen source changed: {path}")
            result[split].append(replace(record, source_group=item["group"]))
    return result


def check_evaluation_scope(manifest: dict, split: str, found: list[str], *, tune: bool,
                           limit: int, freeze: dict | None = None, weights_sha: str | None = None,
                           manifest_sha: str | None = None, policy: dict | None = None):
    expected = manifest["splits"][split]
    if len(expected) != len(set(expected)) or set(expected) != set(found):
        raise ValueError("Evaluation scope missing/duplicate records")
    if limit < 0:
        raise ValueError("limit must be non-negative")
    role = manifest.get("roles", {}).get(split, split)
    if role == "final_holdout":
        if tune or limit:
            raise ValueError("Final holdout forbids tuning or partial evaluation")
        if not freeze or freeze.get("weights_sha256") != weights_sha or freeze.get("split_sha256") != manifest_sha:
            raise ValueError("Final holdout requires an exact frozen model/split signature")
        if freeze.get("evaluation_policy") != policy:
            raise ValueError("Final holdout requires the frozen evaluation policy")
    if tune and role not in {"dev", "calibration"}:
        raise ValueError("Threshold tuning is permitted only on dev/calibration")
