"""Train the global/local research control on a verified JSON manifest.
Manifest: {"entries": [{"image_path": "...", "mask_path": "...", "label": 0|1}]}.
Paths resolve relative to --root. Masks: white=edited; real images need no mask.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from model import GlobalLocalPrototype

def read_example(row,root,size):
    image_path=root/row['image_path']
    with Image.open(image_path) as im:
        image=np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
    label=int(row['label'])
    if label not in (0,1): raise ValueError('Labels must be 0 or 1')
    if label:
        if not row.get('mask_path'): raise ValueError(f'Missing mask: {image_path}')
        with Image.open(root/row['mask_path']) as im:
            mask=np.asarray(im.convert('L').resize((size,size),Image.Resampling.NEAREST),dtype=np.uint8)>127
        if not mask.any(): raise ValueError(f'Empty edited mask: {image_path}')
    else: mask=np.zeros((size,size),dtype=bool)
    x=torch.from_numpy(image.transpose(2,0,1).copy()).unsqueeze(0)
    y=torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return x,y,torch.tensor([[float(label)]])

def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--root',required=True);a.add_argument('--output',required=True);a.add_argument('--epochs',type=int,default=1);a.add_argument('--size',type=int,default=256);a.add_argument('--seed',type=int,default=42);a.add_argument('--device',default='cpu');a.add_argument('--confirm-mask-white-edited',action='store_true');args=a.parse_args()
    if not args.confirm_mask_white_edited: a.error('Verify mask semantics and pass --confirm-mask-white-edited')
    output=Path(args.output)
    if output.exists(): raise FileExistsError(f'Choose a new output path: {output}')
    if args.epochs<1 or args.size<32: a.error('epochs must be positive and size >= 32')
    torch.manual_seed(args.seed);random.seed(args.seed);root=Path(args.root)
    data=json.loads(Path(args.manifest).read_text(encoding='utf-8')); rows=data['entries']
    if not rows: raise ValueError('Empty manifest')
    device=torch.device(args.device);model=GlobalLocalPrototype(global_size=args.size,tile_size=args.size,stride=args.size).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    output.mkdir(parents=True);history=[]
    for epoch in range(args.epochs):
        random.shuffle(rows);total=0.0;model.train()
        for row in rows:
            x,y,label=read_example(row,root,args.size);x,y,label=x.to(device),y.to(device),label.to(device)
            pred=model(x);loss=F.binary_cross_entropy_with_logits(pred['mask_logits'],y)+F.binary_cross_entropy_with_logits(pred['score_logit'],label)
            opt.zero_grad();loss.backward();opt.step();total+=loss.item()
        mean=total/len(rows);history.append({'epoch':epoch+1,'mean_train_loss':mean});print(f'epoch {epoch+1}: loss={mean:.6f}',flush=True)
    torch.save(model.state_dict(),output/'model_weights.pt');(output/'run.json').write_text(json.dumps({'args':vars(args),'count':len(rows),'history':history,'warning':'Training loss only; no accuracy or localization performance measured.'},indent=2),encoding='utf-8')
    print(f'Saved {output.resolve()}')
if __name__=='__main__': main()
