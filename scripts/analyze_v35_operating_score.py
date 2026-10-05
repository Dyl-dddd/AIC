"""Audit full operating-score curves on frozen V12 group-OOF predictions.

Uses image-count mixture weights rather than averaging two view scores.
No Test prediction is changed. Threshold selection is exploratory unless its
held-out-group result improves. The ranker OOF is pre-existing, so this is not a
fully nested refit of the ranker.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_final_submission_postprocess import PERMITTED, load_cell
from steel_defect.metrics import _match_class

GRID = np.unique(np.r_[0., np.geomspace(1e-5, .99, 121), np.linspace(.05, .95, 37)])


def ap(tp, weight, total):
    if total <= 0:
        return float('nan')
    keep = tp > 0
    if not np.any(keep):
        return 0.
    recalls = np.cumsum(tp * weight)[keep] / total
    precision = np.cumsum(tp * weight)[keep] / np.cumsum(weight)[keep]
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    positions = np.searchsorted(recalls, np.linspace(0, 1, 101), side='left')
    return float(np.r_[envelope, 0.][positions].mean())


def load():
    splits = json.loads((ROOT / 'data/official_v2_20260831b/splits.json').read_text(encoding='utf-8'))
    groups = {name: str(row['group']) for name, row in splits['records'].items()}
    fold = lambda name: int.from_bytes(hashlib.sha256(groups[name.split('::')[0]].encode()).digest()[:8], 'big') % 5
    cells = ROOT / 'runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells'
    base = ROOT / 'runs/semifinal/v20_crop_verifier_20261002/v12_oof'
    data = {name: {'scores': [], 'tp': [], 'weight': [], 'fold': [], 'view': [], 'gt': np.zeros((2, 5))} for name in PERMITTED}
    for view_id, (view, count) in enumerate((('original', 488), ('grid_crops', 300))):
        gt = load_cell(cells / f'model_a_tta_s1024_{view}')['ground_truth']
        rows = json.loads((base / f'v12_{view}_predictions.json').read_text(encoding='utf-8'))
        weight = count / len(gt)
        for name in PERMITTED:
            selected = sorted((row for row in rows if row['category_name'] == name), key=lambda row: row['score'], reverse=True)
            tp, _, _ = _match_class(rows, gt, name, .5)
            d = data[name]
            d['scores'].extend(row['score'] for row in selected)
            d['tp'].extend(tp)
            d['weight'].extend([weight] * len(selected))
            d['fold'].extend(fold(row['image_id']) for row in selected)
            d['view'].extend([view_id] * len(selected))
            for image_id, classes in gt.items():
                d['gt'][view_id, fold(image_id)] += len(classes.get(name, [])) * weight
        print(f'Loaded {view}: {len(gt)} images, {len(rows)} predictions', flush=True)
    for d in data.values():
        order = np.argsort(-np.asarray(d['scores']), kind='stable')
        for key in ('scores', 'tp', 'weight', 'fold', 'view'):
            d[key] = np.asarray(d[key])[order]
    return data


def evaluate(data, thresholds, folds=tuple(range(5)), view=None):
    classes = {}
    for name, d in data.items():
        select = np.isin(d['fold'], folds)
        if view is not None:
            select &= d['view'] == view
        t = thresholds.get(name, 0.) if isinstance(thresholds, dict) else thresholds
        select &= d['scores'] >= t
        gt = float(d['gt'][:, folds].sum() if view is None else d['gt'][view, list(folds)].sum())
        w, tp = d['weight'][select], d['tp'][select]
        classes[name] = {'tp': float((tp*w).sum()), 'predictions': float(w.sum()), 'gt': gt, 'ap50': ap(tp, w, gt)}
    tp = sum(d['tp'] for d in classes.values())
    num = sum(d['predictions'] for d in classes.values())
    gt = sum(d['gt'] for d in classes.values())
    p, r = tp / max(num, 1e-12), tp / max(gt, 1e-12)
    m = float(np.nanmean([d['ap50'] for d in classes.values()]))
    return {'score': 20*p+60*r+20*m, 'precision': p, 'recall': r, 'map50': m, 'tp': tp, 'predictions': num, 'gt': gt, 'classes': classes}


def compact(result):
    return {key: value for key, value in result.items() if key != 'classes'}


def main():
    data = load()
    baseline = evaluate(data, 0.)
    curve = [{'threshold': float(t), **compact(evaluate(data, float(t)))} for t in GRID]
    best = max(curve, key=lambda row: row['score'])
    print('BASELINE', compact(baseline), flush=True)
    print('GLOBAL BEST', best, flush=True)
    # Fit one shared threshold on four folds, check it on the excluded fold.
    cv = []
    for fold in range(5):
        train_folds = tuple(i for i in range(5) if i != fold)
        best_t = max(GRID, key=lambda t: evaluate(data, float(t), train_folds)['score'])
        before = evaluate(data, 0., (fold,))
        after = evaluate(data, float(best_t), (fold,))
        cv.append({'fold': fold, 'threshold': float(best_t), 'before': compact(before), 'after': compact(after), 'delta_score': after['score']-before['score']})
        print('FOLD', fold, cv[-1]['threshold'], cv[-1]['delta_score'], flush=True)
    report = {'protocol': 'Fixed 488:300 image-count weighting; V12 existing group-OOF; shared-threshold heldout groups; exploratory, not nested ranker refit',
              'baseline': baseline, 'global_best': best, 'curve': curve, 'heldout_folds': cv,
              'view_checks': {str(view): {'before': compact(evaluate(data, 0., view=view)), 'after': compact(evaluate(data, best['threshold'], view=view))} for view in (0,1)}}
    out = ROOT / 'runs/semifinal/v35_operating_score_20261004/results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out, flush=True)


if __name__ == '__main__':
    main()
