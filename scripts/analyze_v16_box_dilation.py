"""Screen class-specific box dilation on frozen development views only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7


def dilate(rows: list[dict], sizes: dict, class_name: str, sx: float, sy: float) -> list[dict]:
    output = []
    for row in rows:
        if row["category_name"] != class_name:
            output.append(row)
            continue
        x1, y1, x2, y2 = map(float, row["bbox"])
        width, height = sizes[row["image_id"]]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w, half_h = (x2 - x1) * sx / 2, (y2 - y1) * sy / 2
        box = [
            max(0, min(width - 1, round(cx - half_w))),
            max(0, min(height - 1, round(cy - half_h))),
            max(1, min(width, round(cx + half_w))),
            max(1, min(height, round(cy + half_h))),
        ]
        box[2] = max(box[0] + 1, box[2])
        box[3] = max(box[1] + 1, box[3])
        changed = dict(row)
        changed["bbox"] = box
        output.append(changed)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {}
    for view in ("original", "grid_crops"):
        a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        data[view] = (a, fuse_v7(a, b))
        print(f"LOAD {view}: {len(data[view][1])}", flush=True)
    configs = [("baseline", "", 1.0, 1.0)]
    for class_name in ("zonglie", "jieba", "jiaza", "yiwuyaru"):
        for factor in (1.05, 1.10, 1.20):
            configs.append((f"{class_name}_width_{factor:.2f}", class_name, factor, 1.0))
    for factor in (1.05, 1.10):
        configs.append((f"zonglie_both_{factor:.2f}", "zonglie", factor, factor))
    report = {"experiments": {}}
    for name, class_name, sx, sy in configs:
        metrics = {}
        for view, (cell, rows) in data.items():
            changed = rows if name == "baseline" else dilate(rows, cell["sizes"], class_name, sx, sy)
            result = metrics50(changed, cell["ground_truth"], cell["sizes"])
            metrics[view] = {key: result[key] for key in ("score", "map50", "tp", "fp")}
        report["experiments"][name] = metrics
        print(name, {view: (round(m["score"], 6), m["tp"]) for view, m in metrics.items()}, flush=True)
    baseline = report["experiments"]["baseline"]
    for metrics in report["experiments"].values():
        for view, values in metrics.items():
            values["delta_score"] = values["score"] - baseline[view]["score"]
            values["delta_tp"] = values["tp"] - baseline[view]["tp"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
