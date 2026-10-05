"""Train-only, class-conditioned proposal quality model, then independent dev audit.

The target is IoU and IoU>=0.5, not merely crop category. No Test data is read.
All loaders use zero workers. Scores/boxes are never overwritten in place.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small

CLASSES = ('jieba', 'zonglie', 'jiaza', 'yiwuyaru', 'huashang', 'mamianmakeng', 'yanghuatiepi', 'gunyin')
SIZE = 160
SEED = 20261004


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def gray(path):
    result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if result is None:
        raise ValueError(f'Cannot read {path}')
    return result


def ious(box, boxes):
    if not len(boxes):
        return np.zeros(0, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    b = np.asarray(box, dtype=np.float32)
    wh = np.maximum(0, np.minimum(b[2:], boxes[:, 2:]) - np.maximum(b[:2], boxes[:, :2]))
    inter = wh[:, 0] * wh[:, 1]
    area = (b[2]-b[0])*(b[3]-b[1]) + (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1]) - inter
    return inter / np.maximum(area, 1e-8)


def render(image, box):
    h, w = image.shape
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2-x1, 2), max(y2-y1, 2)
    px, py = max(8, bw*.4), max(8, bh*.4)
    l, t = max(0, int(x1-px)), max(0, int(y1-py))
    r, b = min(w, int(np.ceil(x2+px))), min(h, int(np.ceil(y2+py)))
    if r <= l or b <= t:
        raise ValueError(f'Invalid proposal {box}')
    patch = cv2.resize(image[t:b, l:r], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rel = [(x1-l)/(r-l), (y1-t)/(b-t), (x2-l)/(r-l), (y2-t)/(b-t)]
    # Fixed pixel reference keeps the same proposal geometry on a full image
    # and on a 1387x1516 view. All 2395 source training images are 4096x3000.
    geom = rel + [float(np.log(max(bw/4096, 1e-6))), float(np.log(max(bh/3000, 1e-6)))]
    return patch, geom


def inputs(patches, geometries, device):
    a = np.stack(patches)
    masks = np.zeros_like(a, dtype=np.float32)
    for mask, geom in zip(masks, geometries):
        x1, y1, x2, y2 = np.clip(np.rint(np.asarray(geom[:4])*SIZE), 0, SIZE).astype(int)
        mask[y1:y2, x1:x2] = 1.
    x = torch.from_numpy(a).to(device).float().unsqueeze(1)/255.
    mean = torch.tensor([.485, .456, .406], device=device)[None,:,None,None]
    std = torch.tensor([.229, .224, .225], device=device)[None,:,None,None]
    x = (x.expand(-1,3,-1,-1)-mean)/std
    return torch.cat([x, torch.from_numpy(masks).to(device).unsqueeze(1)], 1)


class Quality(nn.Module):
    def __init__(self, init=None):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        if init is not None:
            state = torch.load(init, map_location='cpu', weights_only=True)
            state = state.get('model', state)
            backbone.load_state_dict({k:v for k,v in state.items() if k.startswith('features.')}, strict=False)
        old = backbone.features[0][0]
        new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        with torch.no_grad():
            new.weight[:,:3].copy_(old.weight)
            new.weight[:,3:].zero_()
        backbone.features[0][0] = new
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Embedding(len(CLASSES),16)
        self.head = nn.Sequential(nn.Linear(576+16+6,128),nn.SiLU(),nn.Dropout(.15),nn.Linear(128,2))
        for block in self.features[1:5]:
            for param in block.parameters():
                param.requires_grad = False

    def forward(self, x, classes, geom):
        feature = self.pool(self.features(x)).flatten(1)
        return self.head(torch.cat([feature, self.embedding(classes), geom],1))


def prepare(args):
    meta = read_json(args.split)
    train_groups = {meta['records'][n]['group'] for n in meta['splits']['train']}
    dev_groups = {meta['records'][n]['group'] for n in meta['splits']['dev']}
    if train_groups & dev_groups:
        raise ValueError('Group leakage')
    out = args.out
    (out/'crops').mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    names = meta['splits']['train'][:args.limit] if args.limit else meta['splits']['train']
    for number,name in enumerate(names):
        rec = meta['records'][name]
        image = gray(args.project/'data/official/train'/name)
        h,w = image.shape
        if (w,h) != (rec['width'],rec['height']):
            raise ValueError(f'Image size mismatch {name}')
        grouped = {c:[a[1:] for a in rec['annotations'] if a[0]==c] for c in CLASSES}
        split = 'val' if int.from_bytes(hashlib.sha256(str(rec['group']).encode()).digest()[:4],'big')%10==0 else 'train'
        examples=[]
        for ann in rec['annotations']:
            if ann[0] not in CLASSES:
                continue
            box=np.asarray(ann[1:],dtype=float)
            bw,bh=box[2:]-box[:2]
            center=(box[2:]+box[:2])/2
            examples.append((box,ann[0]))
            for j in range(7):
                scale=np.exp(rng.uniform(-1.0, .7, 2))
                shift=rng.uniform(-.55,.55,2)*[bw,bh]
                size=np.array([bw,bh])*scale
                # Explicitly include truncated long boxes among proposals.
                if j==0:
                    size=np.array([bw,bh*.35])
                candidate=np.r_[center+shift-size/2,center+shift+size/2]
                candidate[[0,2]]=np.clip(candidate[[0,2]],0,w)
                candidate[[1,3]]=np.clip(candidate[[1,3]],0,h)
                if np.all(candidate[2:]-candidate[:2]>=2):
                    examples.append((candidate,ann[0]))
            examples.append((box,str(rng.choice([c for c in CLASSES if c!=ann[0]]))))
        for _ in range(3):
            bw,bh=np.exp(rng.uniform(np.log(16),np.log(600),2))
            bw,bh=min(bw,w*.8),min(bh,h*.8)
            x,y=rng.uniform(0,w-bw),rng.uniform(0,h-bh)
            examples.append(([x,y,x+bw,y+bh],str(rng.choice(CLASSES))))
        for box,cls in examples:
            quality=float(ious(box,grouped[cls]).max()) if grouped[cls] else 0.
            patch,geom=render(image,box)
            filename=f'{len(rows):07d}.png'
            success,buffer=cv2.imencode('.png',patch)
            if not success:
                raise RuntimeError('Encoding failed')
            buffer.tofile(out/'crops'/filename)
            rows.append({'file':filename,'geom':geom,'class':CLASSES.index(cls),'iou':quality,'split':split,'source':name})
        if (number+1)%200==0:
            print('PREPARE',number+1,len(names),len(rows),flush=True)
    write_json(out/'manifest.json',rows)
    write_json(out/'prepare_report.json',{'source_train_images':len(names),'samples':len(rows),'official_dev_used':False,'test_used':False,'workers':0,'split_sha256':hashlib.sha256(args.split.read_bytes()).hexdigest()})


class Crops(Dataset):
    def __init__(self,root,rows,split):
        self.root=root
        self.rows=[r for r in rows if r['split']==split]
        self.augment=split=='train'
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        patch=gray(self.root/'crops'/row['file'])
        if self.augment:
            # Modest illumination change; no label/box-changing transform.
            patch=np.clip((patch.astype(float)-127)*random.uniform(.8,1.2)+127+random.uniform(-8,8),0,255).astype(np.uint8)
        return patch,np.asarray(row['geom'],np.float32),row['class'],row['iou']


def train(args):
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=read_json(args.out/'manifest.json')
    loaders={s:DataLoader(Crops(args.out,rows,s),batch_size=args.batch,shuffle=s=='train',num_workers=0) for s in ('train','val')}
    if any(len(x.dataset)==0 for x in loaders.values()):
        raise ValueError('Need nonempty train and internal validation groups')
    model=Quality(args.init).to(device)
    optimizer=torch.optim.AdamW([{'params':[p for p in model.features.parameters() if p.requires_grad],'lr':2e-5},
                                {'params':list(model.head.parameters())+list(model.embedding.parameters()),'lr':3e-4}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=float('inf'); history=[]
    for epoch in range(args.epochs):
        start=time.monotonic(); log={'epoch':epoch+1}
        for phase,loader in loaders.items():
            model.train(phase=='train')
            for block in model.features[1:5]:
                block.eval()
            loss_sum=0.; count=0
            for patches,geoms,classes,targets in loader:
                x=inputs(list(patches.numpy()),list(geoms.numpy()),device)
                geoms=geoms.to(device); classes=classes.to(device); targets=targets.float().to(device)
                with torch.set_grad_enabled(phase=='train'),torch.amp.autocast('cuda',enabled=device.type=='cuda'):
                    logits=model(x,classes,geoms)
                    binary=(targets>=.5).float()
                    # Balanced quality learning, using weights fixed a priori.
                    bce=nn.functional.binary_cross_entropy_with_logits(logits[:,0],binary,reduction='none')
                    loss=(bce*(1+2*binary)).mean()+nn.functional.mse_loss(logits[:,1].sigmoid(),targets)
                if phase=='train':
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                loss_sum+=float(loss.detach())*len(targets); count+=len(targets)
            log[phase+'_loss']=loss_sum/count
        log['seconds']=time.monotonic()-start; history.append(log)
        print('EPOCH',log,flush=True)
        state={'model':model.state_dict(),'classes':CLASSES,'size':SIZE,'epoch':epoch+1,'history':history,'protocol':'train-only proposal IoU; independent official dev never fitted'}
        torch.save(state,args.out/'last.pt')
        if log['val_loss']<best:
            best=log['val_loss']; torch.save(state,args.out/'best.pt')
        write_json(args.out/'train_report.json',{'history':history,'best_val_loss':best,'device':str(device),'workers':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','train','all'))
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--split',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--init',type=Path)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--batch',type=int,default=128)
    parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    args.split=args.split or args.project/'data/official_v2_20260831b/splits.json'
    args.out=args.out or args.project/'runs/semifinal/v36_box_quality_20261004'
    args.init=args.init or args.project/'runs/semifinal/v20_crop_verifier_20261002/best.pt'
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode in ('prepare','all'):
        prepare(args)
    if args.mode in ('train','all'):
        train(args)


if __name__=='__main__':
    main()
