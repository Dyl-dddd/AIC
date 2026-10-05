"""Frozen V12 dev-only audit of the train-only proposal quality network."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from v36_box_quality import CLASSES, Quality, gray, render, inputs, ious, read_json, write_json


def metrics(rows, gt):
    per_class={}
    for cls in CLASSES:
        selected=sorted((r for r in rows if r['category_name']==cls),key=lambda r:r['score'],reverse=True)
        matched=defaultdict(set); true=[]
        for row in selected:
            overlaps=ious(row['bbox'],gt[row['image_id']].get(cls,[]))
            order=np.argsort(overlaps)[::-1]
            candidate=next((int(i) for i in order if int(i) not in matched[row['image_id']]),None)
            is_tp=candidate is not None and overlaps[candidate]>=.5
            true.append(float(is_tp))
            if is_tp:
                matched[row['image_id']].add(candidate)
        total=sum(len(classes.get(cls,[])) for classes in gt.values())
        t=np.asarray(true); ct=np.cumsum(t)
        recall=ct/max(total,1); precision=ct/np.maximum(np.arange(1,len(t)+1),1)
        envelope=np.maximum.accumulate(precision[::-1])[::-1] if len(t) else precision
        idx=np.searchsorted(recall,np.linspace(0,1,101))
        ap=float(np.r_[envelope,0.][idx].mean()) if total else float('nan')
        per_class[cls]={'ap50':ap,'tp':int(t.sum()),'gt':total,'predictions':len(t)}
    tp=sum(v['tp'] for v in per_class.values()); total=sum(v['gt'] for v in per_class.values())
    p=tp/max(len(rows),1); r=tp/max(total,1); ap=float(np.nanmean([v['ap50'] for v in per_class.values()]))
    return {'score':20*p+60*r+20*ap,'map50':ap,'tp':tp,'fp':len(rows)-tp,'fn':total-tp,'per_class':per_class}


def ground_truth(args,view):
    if args.ground_truth:
        return read_json(args.ground_truth)[view]
    path=args.project/'runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells'/f'model_a_tta_s1024_{view}'/'candidates.jsonl.gz'
    with gzip.open(path,'rt',encoding='utf-8') as f:
        next(f)
        return {r['image_id']:r['ground_truth'] for r in map(json.loads,f)}


def score(args,rows,model,device):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[(r['image_id'],r['category_name'])].append(i)
    selected=[]
    for indices in groups.values():
        selected.extend(sorted(indices,key=lambda i:rows[i]['score'],reverse=True)[:args.topk])
    sources=defaultdict(list)
    for i in selected:
        sources[rows[i]['image_id'].split('::')[0]].append(i)
    result=np.ones(len(rows),np.float32)
    patches=[]; geoms=[]; labels=[]; positions=[]
    def flush():
        if not patches:
            return
        x=inputs(patches,geoms,device)
        with torch.inference_mode(),torch.amp.autocast('cuda',enabled=device.type=='cuda'):
            logits=model(x,torch.tensor(labels,device=device),torch.tensor(geoms,dtype=torch.float32,device=device))
            probs=logits.float().sigmoid()
            quality=torch.sqrt(probs[:,0]*probs[:,1]).clamp(.01,1.)
        result[positions]=quality.cpu().numpy()
        patches.clear(); geoms.clear(); labels.clear(); positions.clear()
    for n,(source,indices) in enumerate(sorted(sources.items()),1):
        image=gray(args.project/'data/official/train'/source)
        for i in indices:
            r=rows[i]; image_id=r['image_id']; current=image
            if '::' in image_id:
                ox,oy=(int(x[1:]) for x in image_id.split('::')[1].split('_'))
                current=image[oy:oy+1516,ox:ox+1387]
            patch,geom=render(current,r['bbox'])
            patches.append(patch); geoms.append(geom); labels.append(CLASSES.index(r['category_name'])); positions.append(i)
            if len(patches)>=args.batch:
                flush()
        if n%100==0:
            print('SCORE',n,len(sources),'selected',len(selected),flush=True)
    flush()
    if not np.isfinite(result).all() or (result<=0).any():
        raise ValueError('Nonfinite quality')
    return result,len(selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path)
    parser.add_argument('--oof',type=Path)
    parser.add_argument('--ground-truth',type=Path)
    parser.add_argument('--topk',type=int,default=3)
    parser.add_argument('--batch',type=int,default=256)
    args=parser.parse_args()
    args.out=args.out or args.project/'runs/semifinal/v36_box_quality_20261004'
    args.oof=args.oof or args.project/'runs/semifinal/v20_crop_verifier_20261002/v12_oof'
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=args.out/'best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if tuple(state['classes'])!=CLASSES:
        raise ValueError('Checkpoint classes differ')
    model=Quality(); model.load_state_dict(state['model']); model.to(device).eval()
    report={'protocol':'Train-only IoU quality model; V12 group-OOF dev, exploratory fixed-alpha policies; no Test access',
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'topk':args.topk,'views':{}}
    for view in ('original','grid_crops'):
        path=args.oof/f'v12_{view}_predictions.json'
        rows=read_json(path); gt=ground_truth(args,view)
        if set(r['image_id'] for r in rows)-set(gt):
            raise ValueError('GT coverage mismatch')
        sources=sorted({r['image_id'].split('::')[0] for r in rows})
        image_stats=[(name,(args.project/'data/official/train'/name).stat().st_size,
                      (args.project/'data/official/train'/name).stat().st_mtime_ns) for name in sources]
        code_hashes=[hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     hashlib.sha256(Path(__file__).with_name('v36_box_quality.py').read_bytes()).hexdigest()]
        signature=hashlib.sha256(json.dumps([report['checkpoint_sha256'],
            hashlib.sha256(path.read_bytes()).hexdigest(),args.topk,gt,image_stats,code_hashes],sort_keys=True).encode()).hexdigest()
        cache=args.out/f'{view}_quality_{signature[:12]}.npy'
        if cache.exists():
            quality=np.load(cache); selected=int((quality!=1).sum())
        else:
            quality,selected=score(args,rows,model,device); np.save(cache,quality)
        if len(quality)!=len(rows):
            raise ValueError('Quality length mismatch')
        baseline=metrics(rows,gt)
        result={'baseline':baseline,'selected_candidates':selected,'policies':{}}
        print('BASELINE',view,baseline['score'],flush=True)
        for alpha in (.1,.25,.5):
            predictions=[{**r,'score':float(r['score']*q**alpha)} for r,q in zip(rows,quality)]
            m=metrics(predictions,gt)
            result['policies'][str(alpha)]={'metrics':m,'score_delta':m['score']-baseline['score'],
                'class_ap_delta':{c:m['per_class'][c]['ap50']-baseline['per_class'][c]['ap50'] for c in CLASSES}}
            print('POLICY',view,alpha,result['policies'][str(alpha)]['score_delta'],result['policies'][str(alpha)]['class_ap_delta'],flush=True)
        report['views'][view]=result
        write_json(args.out/'dev_report.json',report)
    passed=[]
    for alpha in (.1,.25,.5):
        policies=[report['views'][v]['policies'][str(alpha)] for v in report['views']]
        if all(p['score_delta']>.1 for p in policies):
            passed.append(alpha)
    report['decision']={'all_class_dual_view_gain_over_0_1':passed,'official_score_verified':False,
                        'submission_created':False,'note':'Any gain needs group stability and Test distribution checks before deployment.'}
    write_json(args.out/'dev_report.json',report)
    print('DECISION',report['decision'],flush=True)


if __name__=='__main__':
    main()
