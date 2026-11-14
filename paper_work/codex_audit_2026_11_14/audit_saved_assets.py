"""Read-only inspection of existing assets; writes only beside this script."""
import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path('I:/icvgip workshop')
OUT = Path(__file__).parent
result = {'date_local': '2026-11-14 Asia/Calcutta', 'manifests': [], 'errors': []}
all_rows = []
paths = sorted((ROOT/'baseline_work/data_manifests').glob('*.json'))
paths.append(ROOT/'echoseam_research/pilot_train.json')
for path in paths:
    rows = json.loads(path.read_text(encoding='utf-8-sig'))['entries']
    summary = {'path': str(path), 'records': len(rows), 'labels': dict(Counter(str(r['label']) for r in rows)),
               'unique_source_ids': len({r.get('source_id') for r in rows}), 'missing_files': [],
               'hash_mismatches': [], 'mask_formats': Counter(), 'mask_sizes_mismatched': [],
               'tiny_edited_assuming_white': 0, 'nonbinary_masks': 0, 'pairs_original_field_present': 0}
    for row in rows:
        item = dict(row, manifest=str(path))
        image_path = ROOT/row['image_path']
        if not image_path.exists():
            summary['missing_files'].append(str(image_path)); continue
        raw_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        item['computed_hash'] = raw_hash
        if row.get('image_sha256') and raw_hash != row['image_sha256']:
            summary['hash_mismatches'].append(row.get('sample_id',str(image_path)))
        with Image.open(image_path) as im:
            size = im.size
        if row.get('original_path') or row.get('original_image_path'):
            summary['pairs_original_field_present'] += 1
        mask_path = ROOT/row['mask_path'] if row.get('mask_path') else None
        if mask_path:
            if not mask_path.exists():
                summary['missing_files'].append(str(mask_path))
            else:
                with Image.open(mask_path) as mask:
                    summary['mask_formats'][mask.format] += 1
                    if mask.size != size: summary['mask_sizes_mismatched'].append(str(mask_path))
                    a = np.asarray(mask.convert('L'))
                unique = np.unique(a)
                if not set(unique.tolist()).issubset({0,255}): summary['nonbinary_masks'] += 1
                fraction = float((a > 127).mean())
                if row['label'] == 1 and 0 < fraction < .02: summary['tiny_edited_assuming_white'] += 1
        all_rows.append(item)
    summary['mask_formats'] = dict(summary['mask_formats'])
    result['manifests'].append(summary)

cross = []
for key in ('computed_hash', 'source_id'):
    groups=defaultdict(list)
    for r in all_rows:
        if r.get(key): groups[r[key]].append(r)
    for value, rows in groups.items():
        manifests=sorted({r['manifest'] for r in rows})
        if len(manifests)>1:
            cross.append({'key':key,'value':value,'manifests':manifests})
result['cross_manifest_matches']=cross
result['overlap_limit']='Exact hashes and declared IDs only; no proof of source independence or near-duplicate exclusion.'

src=ROOT/'controlled_localization/src/controlled_localization/train.py'
text=src.read_text(encoding='utf-8-sig')
tree=ast.parse(text)
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_training_experiment')
result['training_entry_point']={'code_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
    'function_line':fn.lineno,'argument_load_counts':{name:sum(isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id==name for n in ast.walk(fn)) for name in ['manifest_path','schedule_path']},
    'fixture_calls':[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='create_overfit_dataset']}
result['saved_run_summaries']={}
for variant in ['t0','t1','t2','t2o']:
    p=ROOT/f'controlled_localization/runs/run_{variant}/run_summary.json'
    if p.exists(): result['saved_run_summaries'][variant]=json.loads(p.read_text())

p=ROOT/'controlled_localization/reports/opensdid_per_image.csv'
rows=list(csv.DictReader(p.open(newline='',encoding='utf-8-sig')))
fake=[r for r in rows if int(r['label'])==1]
real=[r for r in rows if int(r['label'])==0]
tiny=[r for r in fake if 0<float(r['edit_pct'])<2]
result['recomputed_saved_benchmark']={
    'scope':'Arithmetic from saved CSV, not fresh inference or independent validation of masks/protocol.',
    'n':len(rows),'edited':len(fake),'genuine':len(real),'tiny':len(tiny),
    'auroc':roc_auc_score([int(r['label']) for r in rows],[float(r['raw_score']) for r in rows]),
    'mean_edited_iou':float(np.mean([float(r['loc_iou']) for r in fake])),
    'mean_tiny_iou':float(np.mean([float(r['loc_iou']) for r in tiny])),
    'mean_genuine_flagged_pixel_pct':float(np.mean([float(r['clean_fp_pct']) for r in real])),
    'genuine_image_positive_fraction':float(np.mean([int(r['is_fake_pred']) for r in real])),
    'edited_image_positive_fraction':float(np.mean([int(r['is_fake_pred']) for r in fake]))}

evidence={}
for rel, terms in {
 'train.py':['create_overfit_dataset','manifest_path','schedule_path','validation_macro_iou','evaluate_dataset(manifest_records','outside_eligible','torch.save'],
 'overfit_check.py':['rng.rand','split="train"','source_dataset="overfit_fixture"'],
 'data/pairs.py':['resize((w, h)','outside_pair_eligible: bool = True'],
 'eval_benchmark_comparison.py':['default=0.50','bootstrap_ci','baseline-metrics'],
}.items():
    p=ROOT/'controlled_localization/src/controlled_localization'/rel
    evidence[rel]=[{'line':i,'text':s.strip()} for i,s in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1) if any(t in s for t in terms)]
result['code_evidence']=evidence
(OUT/'asset_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'manifests':result['manifests'],'cross_manifest_matches':len(cross),'training_entry_point':result['training_entry_point'],'benchmark':result['recomputed_saved_benchmark']},indent=2))
