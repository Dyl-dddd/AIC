"""V39: numeric-source-only adaptation of the strong steel detector.

Train images are from the frozen official training split only. The numeric
development split is used only for checkpoint selection. No Test labels or
pseudo-labels are loaded. This script is a local diagnostic, not a submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import ultralytics
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from delivery.v34_single_detector_adapt.run_v34 import prepare, sha256, SOURCE_SHA256, SPLIT_SHA256


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')


def numeric_split():
    path=ROOT/'data/official_v2_20260831b/splits.json'
    if sha256(path)!=SPLIT_SHA256:
        raise RuntimeError('Frozen split checksum mismatch')
    meta=json.loads(path.read_text(encoding='utf-8'))
    selected={part:[n for n in meta['splits'][part] if n[0].isdigit()] for part in ('train','dev')}
    if not selected['train'] or not selected['dev']:
        raise RuntimeError('Missing numeric source in frozen split')
    meta['splits']={**meta['splits'],**selected}
    print('NUMERIC_SPLIT', {k:len(v) for k,v in selected.items()},flush=True)
    return meta


def evaluate(weight,data_yaml,work,name):
    report_path=work/'validation'/f'{name}.json'
    if report_path.exists():
        prior=json.loads(report_path.read_text(encoding='utf-8'))
        if prior.get('checkpoint_sha256')==sha256(weight):
            return prior
    result=YOLO(str(weight)).val(data=str(data_yaml),split='val',imgsz=1024,batch=1,
                                 workers=0,device=0,conf=.001,iou=.7,max_det=1000,
                                 plots=False,project=str(work/'validation'),name=name,exist_ok=True)
    box=result.box
    classes={result.names[int(c)]:float(ap) for c,ap in zip(box.ap_class_index,box.ap50)}
    report={'checkpoint':str(weight),'checkpoint_sha256':sha256(weight),
            'numeric_dev_map50':float(box.map50),'numeric_dev_map50_95':float(box.map),
            'precision':float(box.mp),'recall':float(box.mr),'class_ap50':classes,
            'protocol':'Frozen numeric-source dev, full+six grid views; identical for source/candidate',
            'official_score_unverified':True}
    write_json(report_path,report)
    print('VAL',name,report['numeric_dev_map50'],flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--epochs',type=int,default=5)
    args=parser.parse_args()
    if ultralytics.__version__!='8.3.169' or not torch.cuda.is_available():
        raise RuntimeError('Expected Ultralytics 8.3.169 and a CUDA GPU')
    source=ROOT/'analysis/full_retrain_v3_fix1_20260925/runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt'
    if not source.exists() or sha256(source)!=SOURCE_SHA256:
        raise RuntimeError('Source weight missing or changed')
    work=ROOT/'runs/semifinal/v39_numeric_adapt'
    work.mkdir(parents=True,exist_ok=True)
    split=numeric_split()
    prepared=prepare(ROOT,work,split,False)
    write_json(work/'protocol.json',{'source_sha256':SOURCE_SHA256,'split_sha256':SPLIT_SHA256,
                                     'numeric_train_images':len(split['splits']['train']),
                                     'numeric_dev_images':len(split['splits']['dev']),
                                     'prepared':prepared,'workers':0})
    if args.prepare_only:
        print('PREPARED',prepared['data_yaml'],flush=True)
        return
    stage=work/'training/numeric_finetune_stable'
    last=stage/'weights/last.pt'
    if last.exists() and not (stage/'COMPLETE.json').exists():
        print('RESUME',last,flush=True)
        YOLO(str(last)).train(resume=True,workers=0)
    elif not (stage/'COMPLETE.json').exists():
        YOLO(str(source)).train(data=prepared['data_yaml'],epochs=args.epochs,imgsz=1024,
            batch=1,device=0,workers=0,project=str(work/'training'),name='numeric_finetune_stable',
            exist_ok=False,optimizer='AdamW',lr0=2e-5,lrf=.2,warmup_epochs=0.,warmup_bias_lr=0.,
            patience=args.epochs+1,amp=True,cache=False,val=False,plots=False,
            save=True,save_period=2,mosaic=0.,mixup=0.,cutmix=0.,
            hsv_h=0.,hsv_s=0.,hsv_v=.05,degrees=0.,translate=.02,
            scale=.04,fliplr=.5,flipud=0.,seed=39,deterministic=True)
    if not last.exists():
        raise RuntimeError('Training failed before last checkpoint')
    write_json(stage/'COMPLETE.json',{'last_sha256':sha256(last),'workers':0})
    source_report=evaluate(source,Path(prepared['data_yaml']),work,'source')
    candidates=[evaluate(path,Path(prepared['data_yaml']),work,path.stem)
                for path in sorted((stage/'weights').glob('epoch*.pt'))+[last]]
    best=max(candidates,key=lambda row:row['numeric_dev_map50'])
    decision={'best_checkpoint':best['checkpoint'],
              'numeric_dev_map50_delta':best['numeric_dev_map50']-source_report['numeric_dev_map50'],
              'passes_initial_gate':best['numeric_dev_map50']>=source_report['numeric_dev_map50']+.02,
              'official_score_unverified':True,'submission_created':False}
    write_json(work/'RESULT.json',{'source':source_report,'candidates':candidates,'decision':decision})
    print('RESULT',json.dumps(decision),flush=True)


if __name__=='__main__':
    main()
