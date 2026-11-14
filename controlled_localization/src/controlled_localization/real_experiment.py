"""Reproducible unpaired real-image development experiment; no synthetic fallback."""
import argparse, csv, hashlib, io, json, random, time, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score
from .models.detector import ControlledLocalizationDetector
from .losses.segmentation import compute_supervised_segmentation_loss
from .losses.consistency import compute_directed_consistency_loss
from .data.canonicalize import apply_recipe
from .infer import run_single_image_inference
from .policy import InferencePolicy
from .utils import seed_all

ROOT=Path('I:/icvgip workshop')
BASE=ROOT/'controlled_localization/runs/real_development_20261114'

def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:return r.read()
        except Exception:
            if attempt==3:raise
            time.sleep(2*(attempt+1))

def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    manifest=BASE/'manifest.json'
    if manifest.exists():return json.loads(manifest.read_text())
    source=get('https://raw.githubusercontent.com/iamwangyabin/OpenSDI/main/data/hf_datasets.py')
    (BASE/'official_mask_loader.py.txt').write_bytes(source)
    excluded=set()
    for p in (ROOT/'baseline_work/data_manifests').glob('*.json'):
        excluded.update(str(r['source_id']) for r in json.loads(p.read_text())['entries'])
    excluded.update(str(r['source_id']) for r in json.loads((ROOT/'echoseam_research/pilot_train.json').read_text())['entries'])
    rows=[]
    # Distinct blocks from the official training split, never old test inputs.
    for offset in [2200,4200,6200,8200,102200,104200,106200,108200]:
        q=urllib.parse.urlencode(dict(dataset='nebula/OpenSDI_train',config='default',split='sd15',offset=offset,length=40))
        batch=json.loads(get('https://datasets-server.huggingface.co/rows?'+q))['rows']
        rows.extend(batch)
        print('Metadata',offset,len(batch),flush=True)
    def download(obj):
        r=obj['row']; sid=Path(r['key']).stem
        if sid in excluded or ('partial' not in r['key']):return None
        label=int(r['label']); name=str(obj['row_idx']); folder=BASE/'data';folder.mkdir(exist_ok=True)
        raw=get(r['image']['src']); image=Image.open(io.BytesIO(raw)).convert('RGB')
        p=folder/(name+'.jpg');p.write_bytes(raw)
        mask=np.zeros((image.height,image.width),np.uint8)
        mp=None
        if label:
            if not r.get('mask'):return None
            mr=get(r['mask']['src']);(folder/(name+'_mask_original.bin')).write_bytes(mr)
            m=Image.open(io.BytesIO(mr)).convert('RGB')
            if m.size!=image.size:raise ValueError('Mask geometry mismatch')
            mask=(np.asarray(m).mean(2)>127.5).astype(np.uint8)
            if mask.sum()==0:return None
            mp=folder/(name+'_mask.png');Image.fromarray(mask*255).save(mp)
        return dict(id=name,source_id=sid,label=label,image=str(p),mask=str(mp) if mp else None,
                    sha256=hashlib.sha256(raw).hexdigest(),fraction=float(mask.mean()),
                    key=r['key'],scope='viewer-derived development; official mask polarity verified')
    with ThreadPoolExecutor(max_workers=6) as ex:
        records=[r for r in ex.map(download,rows) if r]
    # Group by source; deterministic held-out validation. Deduplicate images.
    seen=set(); records=[r for r in records if not (r['sha256'] in seen or seen.add(r['sha256']))]
    for r in records:r['split']='val' if int(hashlib.sha256(r['source_id'].encode()).hexdigest()[:8],16)%5==0 else 'train'
    assert not ({r['source_id'] for r in records if r['split']=='train'} & {r['source_id'] for r in records if r['split']=='val'})
    manifest.write_text(json.dumps(records,indent=2))
    print('Data ready',len(records),{s:sum(r['split']==s for r in records) for s in ['train','val']},flush=True)
    return records

def load(r):
    im=Image.open(r['image']).convert('RGB')
    m=np.zeros((im.height,im.width),np.float32) if not r['mask'] else np.asarray(Image.open(r['mask']),dtype=np.float32)/255
    return im,m

def evaluate(model,records,out,step):
    model.eval();rows=[]
    policy=InferencePolicy(policy_name='real_development_fixed_05',t_mask=.5,t_image=.5)
    for r in records:
        result=run_single_image_inference(r['image'],model,policy,torch.device('cuda'))
        p=result['merged_prob'];_,m=load(r);y=m>.5
        pred=p>.5;inter=(pred&y).sum();union=(pred|y).sum()
        rows.append(dict(id=r['id'],source_id=r['source_id'],label=r['label'],score=float(result['raw_score']),
                         iou=float(inter/union) if union else None,fp_area=float(pred.mean()),fraction=r['fraction']))
    pos=[r for r in rows if r['label']];neg=[r for r in rows if not r['label']]
    metric=dict(step=step,n=len(rows),auroc=float(roc_auc_score([r['label'] for r in rows],[r['score'] for r in rows])),
                mean_iou=float(np.mean([r['iou'] for r in pos])),genuine_positive_area=float(np.mean([r['fp_area'] for r in neg])))
    (out/f'validation_{step}.json').write_text(json.dumps(dict(metrics=metric,predictions=rows),indent=2))
    print('VALIDATION',metric,flush=True);model.train();return metric

def train(records,variant,steps,run_name=None,initialize=None):
    seed_all(42);rng=np.random.default_rng(42);torch.set_num_threads(4)
    out=BASE/(run_name or variant+'_real_v2');out.mkdir(exist_ok=False)
    model=ControlledLocalizationDetector(pretrained_rgb=True).cuda()
    if initialize:
        model.load_state_dict(torch.load(initialize,map_location='cuda',weights_only=False)['model_state_dict'])
    opt=torch.optim.AdamW([{'params':[p for p in model.rgb_encoder.parameters() if p.requires_grad],'lr':1e-4},
                          {'params':list(model.res_encoder.parameters())+list(model.decoder.parameters()),'lr':3e-4}],weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda');trainrows=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val']
    cache={r['id']:load(r) for r in trainrows};classes=[[r for r in trainrows if r['label']==label] for label in [0,1]]
    (out/'config.json').write_text(json.dumps(dict(variant=variant,steps=steps,seed=42,tile=384,
        initialization=initialize,protocol='v1.2 unpaired within-image degradation; outside loss disabled',manifest_sha256=hashlib.sha256((BASE/'manifest.json').read_bytes()).hexdigest()),indent=2))
    best=-1.;log=(out/'train.jsonl').open('w');start=time.time();model.train()
    for step in range(1,steps+1):
        xs=[];ys=[]
        for label in [0,1]:
            r=classes[label][int(rng.integers(len(classes[label])))];im,m=cache[r['id']]
            recipe=['identity','jpeg_q70','jpeg_q85','jpeg_q95','blur_s05'][int(rng.integers(5))]
            processed,_=apply_recipe(im,recipe)
            a=np.asarray(im,dtype=np.float32)/255;b=np.asarray(processed,dtype=np.float32)/255
            h,w=m.shape;ph=max(0,384-h);pw=max(0,384-w)
            a=np.pad(a,((0,ph),(0,pw),(0,0)),mode='edge');b=np.pad(b,((0,ph),(0,pw),(0,0)),mode='edge');m=np.pad(m,((0,ph),(0,pw)))
            h,w=m.shape
            if label and rng.random()<.75:
                pts=np.argwhere(m>.5);cy,cx=pts[int(rng.integers(len(pts)))];x=int(np.clip(cx-192,0,w-384));y=int(np.clip(cy-192,0,h-384))
            else:x=int(rng.integers(w-384+1));y=int(rng.integers(h-384+1))
            for image in [a,b]:xs.append(image[y:y+384,x:x+384].transpose(2,0,1));ys.append(m[y:y+384,x:x+384][None])
        x=torch.tensor(np.stack(xs),device='cuda');y=torch.tensor(np.stack(ys),device='cuda')
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            z=model(x)['mask_logits'];seg,_,_=compute_supervised_segmentation_loss(z,y)
            consistency=compute_directed_consistency_loss(z[1::2],z[0::2],y[0::2]) if variant=='T2' else z.sum()*0
            loss=seg+.1*min(1,step/100)*consistency
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update()
        log.write(json.dumps(dict(step=step,loss=float(loss),seg=float(seg),consistency=float(consistency)))+'\n');log.flush()
        if step%25==0:print(variant,step,'loss',round(float(loss),4),'seconds',int(time.time()-start),flush=True)
        if step%100==0 or step==steps:
            metric=evaluate(model,val,out,step)
            checkpoint=dict(model_state_dict=model.state_dict(),optimizer=opt.state_dict(),step=step,variant=variant)
            torch.save(checkpoint,out/'latest.pt')
            if metric['mean_iou']>best:best=metric['mean_iou'];torch.save(checkpoint,out/'best.pt')
    log.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=400);parser.add_argument('--variant',default='T1',choices=['T1','T2']);parser.add_argument('--run-name');parser.add_argument('--initialize');args=parser.parse_args()
    records=prepare();train(records,args.variant,args.steps,args.run_name,args.initialize)
