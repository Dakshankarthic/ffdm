"""Summarize actual development runs and render predictions from a selected checkpoint."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from .real_experiment import BASE,load
from .models.detector import ControlledLocalizationDetector
from .policy import InferencePolicy
from .infer import run_single_image_inference
from .utils import create_heatmap_overlay

def main():
    candidates=[]
    for p in BASE.glob('*/validation_*.json'):
        j=json.loads(p.read_text());candidates.append(dict(run=p.parent.name,path=str(p),**j['metrics']))
    candidates.sort(key=lambda x:x['mean_iou'],reverse=True)
    best=candidates[0];run=BASE/best['run'];j=json.loads(Path(best['path']).read_text());rows=j['predictions']
    negatives=np.array([r['score'] for r in rows if not r['label']]);positives=np.array([r['score'] for r in rows if r['label']])
    threshold=float(np.sort(negatives)[-2]) # at most one positive of 32 with strict >
    best['validation_selected_image_threshold']=threshold
    best['validation_image_fpr']=float((negatives>threshold).mean())
    best['validation_image_sensitivity']=float((positives>threshold).mean())
    (BASE/'RESULTS.json').write_text(json.dumps(dict(selected=best,all_checkpoints=candidates,
        limitation='Development selection on 66 images; not independent test accuracy. Two tiny validation edits.'),indent=2))
    model=ControlledLocalizationDetector(pretrained_rgb=False).cuda()
    model.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['model_state_dict'])
    policy=InferencePolicy(policy_name='development_selected',t_mask=.5,t_image=threshold)
    (run/'inference_policy.json').write_text(json.dumps(vars(policy),indent=2))
    records=json.loads((BASE/'manifest.json').read_text());valid=[r for r in records if r['split']=='val']
    selected=[r for r in valid if r['label']][:3]+[r for r in valid if not r['label']][:1]
    canvas=Image.new('RGB',(900,4*260),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(selected):
        im,m=load(r);p=run_single_image_inference(r['image'],model,policy,torch.device('cuda'))['merged_prob']
        np.save(run/(r['id']+'_probability.npy'),p)
        for k,pic in enumerate([im,Image.fromarray((m*255).astype('uint8')).convert('RGB'),create_heatmap_overlay(im,p)]):
            pic.thumbnail((295,225));canvas.paste(pic,(k*300,i*260+25))
        draw.text((5,i*260+5),f"{r['id']}: image | ground truth | predicted region",fill='black')
    canvas.save(BASE/'validation_examples.png')
    lines=['REAL-IMAGE DEVELOPMENT RESULTS — 2026-11-14','254 training images; 66 source-ID-disjoint validation images.','']
    for r in candidates:lines.append(f"{r['run']} step {r['step']}: IoU {r['mean_iou']:.4f}, AUROC {r['auroc']:.4f}, genuine positive area {r['genuine_positive_area']:.4f}")
    lines+=['',f'Selected checkpoint: {run / "best.pt"}',f'Validation-selected image threshold {threshold:.6f}; image FPR {best["validation_image_fpr"]:.4f}; sensitivity {best["validation_image_sensitivity"]:.4f}.',
        'These are development results, not independent test accuracy or proof of high accuracy.',
        'Viewer-derived image/mask encoding, limited source diversity, only two tiny validation edits, one seed, and repeated validation selection limit conclusions.',
        'T1_extended is a separate longer fine-tuning run, not an equal-budget T1/T2 comparison.']
    (BASE/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(best,indent=2))
if __name__=='__main__':main()
