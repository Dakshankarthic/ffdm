import json, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from model import GlobalLocalPrototype
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path('echoseam_research/runs/pilot_01')
rows = json.loads(Path('baseline_work/data_manifests/val_sd15_manifest.json').read_text())['entries']
model = GlobalLocalPrototype(global_size=256, tile_size=256, stride=256)
model.load_state_dict(torch.load(run/'model_weights.pt', weights_only=True, map_location='cpu'))
model.cuda().eval()
results = []
example = None
with torch.inference_mode():
    for row in rows:
        im = Image.open(row['image_path']).convert('RGB')
        x = np.asarray(im.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)/255
        pred = model(torch.from_numpy(x).permute(2,0,1)[None].cuda())
        score = pred['score_logit'].sigmoid().item()
        heat = pred['mask_logits'][0,0].sigmoid().cpu().numpy()
        result = dict(sample=row['sample_id'], label=row['label'], score=score, iou=None)
        if row['label']:
            mask = np.asarray(Image.open(row['mask_path']).convert('L').resize((256,256),Image.Resampling.NEAREST))>127
            binary = heat>=.5
            result['iou'] = float((binary & mask).sum() / max(1,(binary | mask).sum()))
            if example is None:
                example = (x,mask,heat,score,result['iou'])
        results.append(result)
with (run/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
summary = dict(n=len(results),accuracy_at_0_5=float(np.mean([(r['score']>=.5)==r['label'] for r in results])),mean_edited_iou_at_0_5=float(np.mean([r['iou'] for r in results if r['iou'] is not None])),scope='Development pilot. Viewer assets, bright-mask assumption, 256x256 metrics. No formal test result.')
(run/'evaluation.json').write_text(json.dumps(summary,indent=2))
x,mask,heat,score,iou=example
fig,axes=plt.subplots(1,3,figsize=(10,3.7),layout='constrained')
for ax,data,title in zip(axes,[x,mask,heat],['Held-out edited image','Dataset mask','Model heatmap (0–1)']):
    ax.imshow(data,cmap='magma',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
fig.suptitle(f'EchoSeam prototype pilot | score {score:.3f} | mask IoU {iou:.3f}')
fig.savefig(run/'held_out_example.png',dpi=145)
print(json.dumps(summary,indent=2))
