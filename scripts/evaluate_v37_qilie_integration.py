"""Evaluate a trained q specialist against frozen V12 original-view OOF.

This is a development proxy only. It does not infer on Test or create a
submission, and should not be interpreted as a leaderboard guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'delivery/v37_qilie_specialist'))

from run_v37 import q_inference, read_image
from scripts.analyze_final_submission_postprocess import load_cell
from steel_defect.classes import CLASS_NAMES
from steel_defect.metrics import _match_class, average_precision


def score(predictions, gt):
    per_class = {}
    for cls in CLASS_NAMES:
        tp, fp, total = _match_class(predictions, gt, cls, .5)
        per_class[cls] = {'tp': int(tp.sum()), 'fp': int(fp.sum()), 'gt': total,
                          'ap50': average_precision(tp, fp, total)}
    tp = sum(r['tp'] for r in per_class.values())
    fp = sum(r['fp'] for r in per_class.values())
    total = sum(r['gt'] for r in per_class.values())
    ap = float(np.nanmean([r['ap50'] for r in per_class.values()]))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total, 1)
    return {'score_proxy': 20*precision+60*recall+20*ap, 'precision': precision,
            'recall': recall, 'map50': ap, 'tp': tp, 'fp': fp, 'gt': total,
            'per_class': per_class}


def candidate_predictions(weights, names, root, cache):
    signature = hashlib.sha256(weights.read_bytes()).hexdigest()
    if cache.exists():
        data = json.loads(cache.read_text(encoding='utf-8'))
        if data['weights_sha256'] == signature and data['image_count'] == len(names):
            return data['predictions']
    model = YOLO(str(weights))
    predictions = []
    for i, name in enumerate(names, 1):
        image = read_image(root / 'data/official/train' / name)
        for confidence, box in q_inference(model, image, 960):
            predictions.append({'image_id': name, 'category_name': 'qilie',
                                'bbox': [int(round(v)) for v in box], 'score': confidence})
        if i % 25 == 0 or i == len(names):
            print('INFERENCE', i, len(names), flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'weights_sha256': signature, 'image_count': len(names),
                                 'predictions': predictions}, ensure_ascii=False), encoding='utf-8')
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    args = parser.parse_args()
    root = ROOT
    cell = load_cell(root / 'runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells/model_a_tta_s1024_original')
    frozen = json.loads((root / 'data/official_v2_20260831b/splits.json').read_text(encoding='utf-8'))
    names = sorted(frozen['splits']['dev'])
    if set(cell['ground_truth']) != set(names):
        raise RuntimeError('Cell and frozen dev image lists differ')
    gt = {}
    for name in names:
        gt[name] = {}
        for cls, *box in frozen['records'][name]['annotations']:
            gt[name].setdefault(cls, []).append([float(v) for v in box])
        for cls in CLASS_NAMES:
            if cls != 'qilie' and sorted(gt[name].get(cls, [])) != sorted(cell['ground_truth'][name].get(cls, [])):
                raise RuntimeError(f'Non-q GT mismatch in {name}: {cls}')
    baseline = json.loads((root / 'runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_original_predictions.json').read_text(encoding='utf-8'))
    if any(row['category_name'] == 'qilie' for row in baseline):
        raise RuntimeError('V12 baseline unexpectedly includes q')
    if set(names) != {row['image_id'] for row in baseline}:
        raise RuntimeError('Baseline does not cover exact frozen dev image set')
    out = root / 'runs/semifinal/v37_qilie_specialist/integration'
    candidates = candidate_predictions(args.weights, names, root, out / f'{args.weights.stem}_q_candidates.json')
    report = {'protocol': 'Existing V12 original-view OOF plus q-only specialist; full frozen dev; nine-class proxy',
              'weights': str(args.weights.resolve()), 'baseline': score(baseline, gt),
              'with_q': score(baseline + candidates, gt), 'q_candidates': len(candidates)}
    report['delta_score_proxy'] = report['with_q']['score_proxy'] - report['baseline']['score_proxy']
    out.mkdir(parents=True, exist_ok=True)
    target = out / f'{args.weights.stem}_integration.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('INTEGRATION', target, report['delta_score_proxy'], flush=True)


if __name__ == '__main__':
    main()
