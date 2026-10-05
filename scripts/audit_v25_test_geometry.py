"""Read-only pixel and geometry audit of Test and train source strata."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/semifinal/v25_test_audit_20261003'

def summary(rows):
    keys = ('mean', 'std', 'black_fraction', 'edge_density', 'column_std', 'row_std')
    return {key: {str(q): float(np.percentile([r[key] for r in rows], q))
                  for q in (10, 50, 90)} for key in keys}

def preview(path, tile_index=None):
    with Image.open(path) as im:
        if tile_index is None:
            im.draft('L', (512, 512))
            im = im.convert('L')
        else:
            x = (0, 1354, 2709)[tile_index % 3]
            y = (0, 1484)[tile_index // 3]
            im = im.convert('L').crop((x, y, x + 1387, y + 1516))
        im.thumbnail((512, 512))
        return np.asarray(im).copy()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ROOT/'data/official_v2_20260831b/splits.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    dimensions = Counter()
    cameras = Counter()
    for path in sorted((ROOT/'data/semifinal/test').glob('*.jpg')):
        with Image.open(path) as im:
            size = im.size
            dimensions[str(size)] += 1
        camera = re.search(r'Raw(\d+)', path.name, re.I)
        cameras[camera.group(1) if camera else 'unknown'] += 1
        groups['test_full' if size == (4096, 3000) else 'test_tile'].append((path, None))
    for family in ('numeric', 'C'):
        names = sorted(n for n in meta['splits']['train'] if n.startswith('C') == (family == 'C'))
        for index in np.linspace(0, len(names)-1, min(200, len(names)), dtype=int):
            path = ROOT/'data/official/train'/names[index]
            groups[f'train_{family}_full'].append((path, None))
            if family == 'numeric':
                groups['train_numeric_tile'].append((path, int(index) % 6))
    stats = {}
    all_rows = {}
    for name, paths in groups.items():
        rows = []
        for i, (path, tile) in enumerate(paths):
            a = preview(path, tile)
            rows.append({'file': path.name, 'mean': float(a.mean()), 'std': float(a.std()),
                         'black_fraction': float((a <= 5).mean()),
                         'edge_density': float((cv2.Canny(a, 50, 150)>0).mean()),
                         'column_std': float(a.mean(axis=0).std()),
                         'row_std': float(a.mean(axis=1).std())})
        all_rows[name] = rows
        stats[name] = {'sample_count': len(rows), 'percentiles': summary(rows)}
        print(name, len(rows), 'mean_median', stats[name]['percentiles']['mean']['50'], flush=True)
    annotations = {}
    for family in ('numeric','C'):
        by_class = defaultdict(list)
        for name in meta['splits']['train']:
            if name.startswith('C') != (family == 'C'):
                continue
            for cls,x1,y1,x2,y2 in meta['records'][name]['annotations']:
                by_class[cls].append((x2-x1,y2-y1))
        annotations[family] = {}
        for cls, boxes in by_class.items():
            a = np.array(boxes)
            small = a.min(axis=1)
            annotations[family][cls] = {'count':len(a), 'median_width':float(np.median(a[:,0])),
                'median_height':float(np.median(a[:,1])),
                'minor_side_below8_at_full1280':float((small*1280/4096<8).mean()),
                'minor_side_below8_at_tile1280':float((small*1280/1516<8).mean())}
    report={'test_dimensions':dict(dimensions),'test_cameras':dict(cameras),'pixel_stats':stats,
            'train_annotation_geometry':annotations,
            'limits':'Test has no ground-truth annotations; prediction size is not true defect size. Pixel previews are bounded to 512 px.'}
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'per_image_stats.json').write_text(json.dumps(all_rows,ensure_ascii=False),encoding='utf-8')
    sheet = Image.new('RGB',(1200,960),'white')
    draw = ImageDraw.Draw(sheet)
    for domain_index, domain in enumerate(('test_full','test_tile')):
        ordered=sorted(all_rows[domain],key=lambda row:row['mean'])
        for j, fraction in enumerate((.05,.20,.40,.60,.80,.95)):
            row=ordered[round(fraction*(len(ordered)-1))]
            a=preview(ROOT/'data/semifinal/test'/row['file'])
            im=Image.fromarray(a).convert('RGB'); im.thumbnail((390,195))
            index=domain_index*6+j; x=(index%3)*400; y=(index//3)*240
            sheet.paste(im,(x,y+35)); draw.text((x+4,y+3),f'{domain} mean={row["mean"]:.1f} std={row["std"]:.1f}',fill='black')
            draw.text((x+4,y+18),row['file'][:45],fill='black')
    sheet.save(OUT/'test_contact_sheet.jpg')
    print('WROTE',OUT,flush=True)

if __name__=='__main__':
    main()
