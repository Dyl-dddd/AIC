"""Audit returned stage1 evidence and run bounded CPU-only postprocessing ablations."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
import zipfile
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_semifinal_stage1 import BASE_SHA, NEW_SHA, SPLIT_SHA, allowed_gt, crop_gt, score_proxy, write_json
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import generate_grid_tiles
from steel_defect.inference import InferenceOptions, merge_candidates
from steel_defect.metrics import _match_class, average_precision, validate_predictions
from steel_defect.voc import parse_voc

PERMITTED = [c for c in CLASS_NAMES if c != "qilie"]


def metrics50(predictions, gt, sizes):
    validate_predictions(predictions, gt, PERMITTED, sizes)
    per_class = {}
    for name in PERMITTED:
        tp, fp, n = _match_class(predictions, gt, name, 0.5)
        per_class[name] = {"gt": n, "tp": int(tp.sum()), "fp": int(fp.sum()),
            "fn": n-int(tp.sum()), "ap50": average_precision(tp, fp, n)}
    n = sum(c["gt"] for c in per_class.values())
    tp = sum(c["tp"] for c in per_class.values())
    fp = sum(c["fp"] for c in per_class.values())
    p, r = tp/max(tp+fp, 1), tp/max(n, 1)
    ap = float(np.mean([c["ap50"] for c in per_class.values() if c["gt"]]))
    return {"p_micro": p, "r_micro": r, "map50": ap, "score": score_proxy(p,r,ap),
        "gt": n, "tp": tp, "fp": fp, "fn": n-tp, "predictions": len(predictions), "per_class": per_class}


def replay(rows, options):
    predictions = []
    for row in rows:
        predictions.extend({"image_id": row["image_id"], **p}
            for p in merge_candidates(row["candidates"], tuple(row["image_size"]), CLASS_NAMES, options)
            if p["category_name"] != "qilie")
    return predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    with zipfile.ZipFile(ROOT/"updates/semifinal_stage1_eval_20260923.zip") as z:
        package = json.loads(z.read("semifinal_stage1_20260923/STAGE1_PACKAGE_MANIFEST.json"))
    split_file=ROOT/"data/official_v2_20260831b/splits.json"
    assert hashlib.sha256(split_file.read_bytes()).hexdigest()==SPLIT_SHA
    manifest=json.loads(split_file.read_text(encoding="utf-8"))
    ids=set(manifest["splits"]["dev"])
    source=ROOT/"data/official/train"
    records={name:parse_voc(source/name,(source/name).with_suffix(".xml")) for name in ids}
    original_gt={name:allowed_gt(r.annotations) for name,r in records.items()}
    grid_gt={f"{name}::x{t.x}_y{t.y}":crop_gt(r,t) for name,r in records.items()
        for t in generate_grid_tiles(r.width,r.height,1387,1516,2,3)}
    preflight=json.loads((args.results/"preflight.json").read_text(encoding="utf-8"))
    assert preflight["scope"]=="complete_dev" and preflight["split_sha256"]==SPLIT_SHA
    assert preflight["baseline_sha256"]==BASE_SHA and preflight["candidate_sha256"]==NEW_SHA
    assert preflight["source_bytes_verified"] and not preflight["training_started"]
    comparison=json.loads((args.results/"comparison.json").read_text(encoding="utf-8"))
    audit, ablations={},{}
    for folder in sorted(p for p in args.results.iterdir() if p.is_dir()):
        name=folder.name
        report=json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
        assert report==comparison[name]
        sig=report["signature"]
        assert report["complete"] and report["sources"]==459 and sig["limit"]==0
        assert sig["split_sha256"]==SPLIT_SHA and set(sig["source_ids"])==ids
        assert sig["weights_sha256"]==(BASE_SHA if name.startswith("old_") else NEW_SHA)
        assert all(package.get(file)==sha for file,sha in sig["code"].items())
        with gzip.open(folder/"candidates.jsonl.gz","rt",encoding="utf-8") as h:
            header=json.loads(next(h)); assert header["signature"]==sig
            rows=[json.loads(line) for line in h]
        gt={row["image_id"]:row["ground_truth"] for row in rows}
        sizes={row["image_id"]:tuple(row["image_size"]) for row in rows}
        assert len(gt)==len(rows)==report["views"]
        assert {row["source_id"] for row in rows}==ids
        expected_gt=original_gt if sig["view"]=="original" else grid_gt
        assert gt==expected_gt, f"GT mismatch: {name}"
        for row in rows:
            r=records[row["source_id"]]
            assert tuple(row["image_size"])==((r.width,r.height) if sig["view"]=="original" else (1387,1516))
        base=InferenceOptions(**sig["options"])
        exported=json.loads((folder/"predictions.json").read_text(encoding="utf-8"))
        reproduced=metrics50(exported,gt,sizes)
        s=report["all_exported"]["summary"]
        assert abs(reproduced["score"]-s["score_proxy_20_60_20"])<1e-9
        assert reproduced["tp"]==s["true_positives"] and reproduced["fp"]==s["false_positives"]
        replayed=replay(rows,base)
        localbase=metrics50(replayed,gt,sizes)
        a=Counter(json.dumps(p,sort_keys=True) for p in replayed)
        b=Counter(json.dumps(p,sort_keys=True) for p in exported)
        difference_count=sum((a-b).values())+sum((b-a).values())
        # Equal-score greedy NMS can select different boxes across NumPy builds.
        # Never silently claim byte-identical replay; compare all ablations to
        # the same local replay and preserve the original uploaded baseline.
        replay_delta=localbase["score"]-reproduced["score"]
        audit[name]={"verified_export_metrics":True,"result":reproduced,"views":len(rows),
            "weights_sha256":sig["weights_sha256"],"local_replay":localbase,
            "replay_symmetric_difference_count":difference_count,"replay_score_delta":replay_delta}
        if abs(replay_delta)>0.1 or abs(localbase["tp"]-reproduced["tp"])>1:
            raise ValueError(f"Material platform replay difference: {name}: {replay_delta}")
        print(f"AUDIT {name}: score={reproduced['score']:.6f}; GT={reproduced['gt']}",flush=True)
        # Only leading candidate and strongest old baseline, both views.
        if name.startswith(("new_semifinal_grid_","old_sliding_")):
            variants={"baseline":base,"nms035":replace(base,iou=0.35),
                "nms070":replace(base,iou=0.70),"nms085":replace(base,iou=0.85),
                "edge085":replace(base,edge_penalty=0.85)}
            ablations[name]={}
            for tag,option in variants.items():
                result = localbase if tag=="baseline" else metrics50(replay(rows,option),gt,sizes)
                result={**result,"delta":result["score"]-localbase["score"],
                    "relative_gain_percent":100*(result["score"]/localbase["score"]-1)}
                ablations[name][tag]=result
                print(f"  {tag}: {result['score']:.6f} ({result['delta']:+.3f}), TP={result['tp']}, FP={result['fp']}",flush=True)
        write_json(args.output/"audit.json",audit)
        write_json(args.output/"postprocess.json",ablations)
    lines=["# 回传包审计与CPU后处理对照","",
        "核对已发布代码哈希、两权重SHA、冻结dev ID、本地XML真值、缓存回放与参考分。云端环境来自回传预检记录，非本机远程实时检查。",
        "没有训练或新GPU推理；阈值维持生成下限0.001。相同分数NMS可能受跨平台排序影响，audit.json记录回放差异；下表变化均相对于同平台回放基线。","",
        "|方案|后处理|P|R|AP50|参考分|变化|TP|FP|","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,variants in ablations.items():
        for tag,v in variants.items():
            lines.append(f"|{name}|{tag}|{v['p_micro']:.5f}|{v['r_micro']:.5f}|{v['map50']:.5f}|{v['score']:.3f}|{v['delta']:+.3f}|{v['tp']}|{v['fp']}|")
    lines.extend(["","单次dev探索，无多种子方差或独立测试证明；不能把局部最优参考分当官方成绩。"])
    (args.output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
