"""Classify high-ranking numerical-family dev errors against existing labels."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scripts.analyze_final_submission_postprocess import load_cell
from steel_defect.geometry import box_iou
from steel_defect.metrics import _match_class

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/semifinal/v25_test_audit_20261003'

def main():
    report={}
    for view in ('original','grid_crops'):
        cell=load_cell(ROOT/f'runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells/model_a_tta_s1024_{view}')
        rows=json.loads((ROOT/f'runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_{view}_predictions.json').read_text(encoding='utf-8'))
        rows=[r for r in rows if not r['image_id'].startswith('C')]
        classes=sorted({r['category_name'] for r in rows})
        report[view]={}
        for cls in classes:
            selected=sorted([r for r in rows if r['category_name']==cls],key=lambda r:r['score'],reverse=True)
            tp,fp,_=_match_class(selected,cell['ground_truth'],cls,.5)
            failures=[]
            counts=Counter()
            for row,is_tp in zip(selected[:100],tp[:100],strict=True):
                if is_tp:
                    counts['TP']+=1; continue
                gt=cell['ground_truth'].get(row['image_id'],{})
                a=np.asarray(row['bbox'],dtype=np.float32)
                own=np.asarray(gt.get(cls,[]),dtype=np.float32)
                other=np.asarray([box for name,boxes in gt.items() if name!=cls for box in boxes],dtype=np.float32)
                same_iou=float(box_iou(a,own).max()) if len(own) else 0.
                other_iou=float(box_iou(a,other).max()) if len(other) else 0.
                kind=('duplicate' if same_iou>=.5 else 'wrong_class' if other_iou>=.5
                      else 'localization' if same_iou>=.1 else 'background_or_unlabeled')
                counts[kind]+=1
                nearest=own[int(box_iou(a,own).argmax())].tolist() if len(own) else None
                failures.append({**row,'kind':kind,'same_iou':same_iou,'other_iou':other_iou,'nearest_gt':nearest})
            loc=[r for r in failures if r['kind']=='localization']
            ratios={}
            if loc:
                ratios={key:float(np.median([(r['bbox'][end]-r['bbox'][start])/(r['nearest_gt'][end]-r['nearest_gt'][start]) for r in loc]))
                        for key,start,end in [('width_ratio',0,2),('height_ratio',1,3)]}
            report[view][cls]={'top100_counts':dict(counts),'localization_median_ratios':ratios,'examples':failures[:5]}
            if view=='original' and cls=='zonglie':
                sheet=Image.new('RGB',(1200,760),'white'); draw=ImageDraw.Draw(sheet)
                for i,row in enumerate(loc[:6]):
                    with Image.open(ROOT/'data/official/train'/row['image_id']) as source:
                        im=source.convert('RGB'); bounds=row['bbox']; gt=row['nearest_gt']
                        x0=max(0,int(min(bounds[0],gt[0])-100)); y0=max(0,int(min(bounds[1],gt[1])-100))
                        x1=min(im.width,int(max(bounds[2],gt[2])+100)); y1=min(im.height,int(max(bounds[3],gt[3])+100))
                        im=im.crop((x0,y0,x1,y1)); d=ImageDraw.Draw(im)
                        for box,color in [(bounds,'red'),(gt,'lime')]:
                            d.rectangle((box[0]-x0,box[1]-y0,box[2]-x0,box[3]-y0),outline=color,width=4)
                        im.thumbnail((390,335)); x=(i%3)*400; y=(i//3)*380
                        sheet.paste(im,(x,y+35)); draw.text((x+3,y+3),f'zonglie IoU={row["same_iou"]:.3f} red=prediction green=GT',fill='black')
                sheet.save(OUT/'zonglie_localization_examples.jpg')
        print(view,{k:v['top100_counts'] for k,v in report[view].items()},flush=True)
    (OUT/'rank_errors.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__': main()
