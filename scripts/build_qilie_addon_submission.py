"""Build a SUBMIT_ONLY package that adds qilie (9th class) boxes to V12.

用途：把云端 Model A 导出的 qilie 测试预测合并进已获官方 67.92 的 V12
submission.json，按保守阈值过滤，逐字保证原 V12 行不变，然后校验并打包。

用法（示例）：
  python scripts/build_qilie_addon_submission.py \
      --v12 "tmp/v12_submit/submission.json" \
      --qilie "runs/semifinal/qilie_test_predictions.json" \
      --test-dir "data/semifinal/test" \
      --threshold 0.30 \
      --out "D:/新建文件夹 (8)/semifinal_v35_qilie_addon_SUBMIT_ONLY.zip"

qilie 文件格式（与 submission 相同的记录字段）：
  [{"image_id": "...jpg", "category_name": "qilie",
    "bbox": [x1, y1, x2, y2], "score": 0.0~1.0}, ...]

自测（无需真实 qilie 文件，用合成数据验证全流程）：
  python scripts/build_qilie_addon_submission.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import zipfile

REQUIRED_FIELDS = ("image_id", "category_name", "bbox", "score")
QILIE = "qilie"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: pathlib.Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"ERROR: {path} 不是非空 JSON 数组")
    return rows


def image_sizes(test_dir: pathlib.Path | None) -> dict[str, tuple[int, int]]:
    if test_dir is None:
        return {}
    sizes: dict[str, tuple[int, int]] = {}
    try:
        from PIL import Image
    except Exception:
        print("WARNING: 未安装 PIL，跳过尺寸/越界校验", file=sys.stderr)
        return {}
    for path in test_dir.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            with Image.open(path) as image:
                sizes[path.name] = image.size
    return sizes


def digest_row(row: dict) -> str:
    payload = json.dumps(
        [row["image_id"], row["category_name"], row["bbox"]],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def filter_qilie(raw: list[dict], threshold: float, v12_ids: set[str],
                 sizes: dict[str, tuple[int, int]]) -> tuple[list[dict], dict]:
    seen: set[tuple[str, tuple[int, ...]]] = set()
    kept: list[dict] = []
    stats = {"input": len(raw), "wrong_class": 0, "below_threshold": 0,
             "bad_bbox": 0, "unknown_image": 0, "duplicate": 0, "kept": 0}
    for row in raw:
        if not all(field in row for field in REQUIRED_FIELDS):
            raise SystemExit(f"ERROR: qilie 记录字段不完整: {row}")
        if row["category_name"] != QILIE:
            stats["wrong_class"] += 1
            continue
        score = float(row["score"])
        if score < threshold:
            stats["below_threshold"] += 1
            continue
        bbox = row["bbox"]
        if len(bbox) != 4 or not all(isinstance(v, int) for v in bbox):
            stats["bad_bbox"] += 1
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            stats["bad_bbox"] += 1
            continue
        image_id = row["image_id"]
        if image_id not in v12_ids:
            stats["unknown_image"] += 1
            continue
        if sizes:
            width, height = sizes[image_id]
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                stats["bad_bbox"] += 1
                continue
        key = (image_id, tuple(bbox))
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({
            "image_id": image_id,
            "category_name": QILIE,
            "bbox": [x1, y1, x2, y2],
            "score": round(min(max(score, 0.0), 1.0), 6),
        })
    stats["kept"] = len(kept)
    return kept, stats


def validate(merged: list[dict], v12: list[dict], sizes: dict[str, tuple[int, int]]) -> None:
    # 1) original V12 rows must be byte-identical and in the same positions
    for index, original in enumerate(v12):
        if merged[index] != original:
            raise SystemExit(f"ERROR: 第 {index} 行 V12 原记录被改动，已中止")
    # 2) field / range checks for the appended qilie rows
    for row in merged[len(v12):]:
        if tuple(sorted(row.keys())) != tuple(sorted(REQUIRED_FIELDS)):
            raise SystemExit(f"ERROR: 字段不符: {row}")
        if not 0.0 <= float(row["score"]) <= 1.0:
            raise SystemExit(f"ERROR: score 越界: {row}")
        if row["category_name"] != QILIE:
            raise SystemExit("ERROR: 追加了非 qilie 行")
    # 3) global checks
    if sizes:
        for row in merged:
            if row["image_id"] not in sizes:
                raise SystemExit(f"ERROR: 未知 image_id: {row['image_id']}")
    print(f"VALIDATE OK: 原 {len(v12)} 行逐字不变，追加 qilie {len(merged) - len(v12)} 行，合计 {len(merged)}")


def build(v12_path: pathlib.Path, qilie_path: pathlib.Path, test_dir: pathlib.Path | None,
          threshold: float, out_path: pathlib.Path) -> dict:
    v12 = load_rows(v12_path)
    raw_qilie = load_rows(qilie_path)
    sizes = image_sizes(test_dir)
    v12_ids = {row["image_id"] for row in v12}
    qilie_rows, stats = filter_qilie(raw_qilie, threshold, v12_ids, sizes)
    if not qilie_rows:
        raise SystemExit("ERROR: 过滤后没有任何 qilie 框，请检查阈值/文件，中止打包")
    merged = v12 + qilie_rows
    validate(merged, v12, sizes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp) / "submission.json"
        staging.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(staging, arcname="submission.json")
    report = {
        "v12_path": str(v12_path),
        "qilie_path": str(qilie_path),
        "threshold": threshold,
        "v12_rows": len(v12),
        "qilie_filter": stats,
        "total_rows": len(merged),
        "out_zip": str(out_path),
        "out_sha256": sha256_file(out_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def self_test() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    v12_path = root / "tmp" / "v12_submit" / "submission.json"
    if not v12_path.exists():
        raise SystemExit("SELF-TEST 需要 tmp/v12_submit/submission.json（由 V12 SUBMIT_ONLY 解压得到）")
    v12 = load_rows(v12_path)
    # fabricate two plausible qilie predictions on two images already present
    targets = [row["image_id"] for row in v12[:2]]
    fake = [
        {"image_id": targets[0], "category_name": QILIE, "bbox": [100, 100, 260, 220], "score": 0.42},
        {"image_id": targets[1], "category_name": QILIE, "bbox": [300, 300, 520, 480], "score": 0.55},
        {"image_id": targets[1], "category_name": "zonglie", "bbox": [1, 1, 20, 20], "score": 0.9},  # ignored
        {"image_id": targets[1], "category_name": QILIE, "bbox": [1, 1, 20, 20], "score": 0.05},      # below thr
    ]
    with tempfile.TemporaryDirectory() as tmp:
        qpath = pathlib.Path(tmp) / "qilie.json"
        qpath.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
        out = pathlib.Path(tmp) / "selftest_qilie.zip"
        report = build(v12_path, qpath, None, 0.30, out)
        assert report["qilie_filter"]["kept"] == 2, report
        # verify the zip re-reads with V12 prefix intact
        with zipfile.ZipFile(out) as archive:
            assert archive.namelist() == ["submission.json"]
            reread = json.loads(archive.read("submission.json").decode("utf-8"))
        assert reread[: len(v12)] == v12
        assert len(reread) == len(v12) + 2
    print("SELF-TEST PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12", type=pathlib.Path, help="V12 submission.json 路径")
    parser.add_argument("--qilie", type=pathlib.Path, help="qilie 测试预测 JSON 路径")
    parser.add_argument("--test-dir", type=pathlib.Path, default=None, help="Test 图像目录（用于越界校验，可选）")
    parser.add_argument("--threshold", type=float, default=0.30, help="qilie 保留阈值（建议 0.25-0.35）")
    parser.add_argument("--out", type=pathlib.Path, help="输出 SUBMIT_ONLY zip 路径")
    parser.add_argument("--self-test", action="store_true", help="用合成数据自测")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not (args.v12 and args.qilie and args.out):
        parser.error("非自测模式需要 --v12、--qilie、--out")
    build(args.v12, args.qilie, args.test_dir, args.threshold, args.out)


if __name__ == "__main__":
    main()
