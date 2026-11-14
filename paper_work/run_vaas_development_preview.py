import os,json,csv,time,hashlib,importlib.metadata
from pathlib import Path
os.environ['HF_HOME']=str(Path('baselines/model_cache').resolve())
import numpy as np,torch
from vaas.inference.pipeline import VAASPipeline
from huggingface_hub import HfApi
p=Path('research_data/opensdi_development_preview');data=json.loads((p/'manifest.json').read_text());out=Path('paper_work/vaas_development_preview');out.mkdir(parents=True,exist_ok=True)
model=VAASPipeline.from_pretrained('OBA-Research/vaas',device='cuda',model_variant='v2-base-df2023'); print('Model loaded',flush=True)
rows=[]
for item in data['items']:
 img=p/item['assets']['image']['file'];t=time.perf_counter();r=model(str(img.resolve()));a=np.asarray(r.pop('anomaly_map'));name=f"row_{item['row_index']:06d}_map.npy";np.save(out/name,a)
 row={'row_index':item['row_index'],'key':item['key'],'label':item['label'],'image':str(img),'S_F':r['S_F'],'S_P':r['S_P'],'S_H':r['S_H'],'runtime_seconds':round(time.perf_counter()-t,3),'map_file':name};rows.append(row);print(f"done {len(rows)}/16 row {row['row_index']}",flush=True)
with (out/'scores.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'n':len(rows),'counts':{str(k):sum(v['label']==k for v in rows) for k in (0,1)},'mean_S_H_by_label':{str(k):sum(v['S_H'] for v in rows if v['label']==k)/sum(v['label']==k for v in rows) for k in (0,1)},'label_mapping':{'0':'real','1':'fake'},'model_repo':'OBA-Research/vaas','model_variant':'v2-base-df2023','device':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),'vaas':importlib.metadata.version('vaas'),'dataset':'nebula/OpenSDI_train','split':'sd15','sample':'16 systematically selected viewer training rows, not random, non-independent test','warning':'exploratory only, viewer may transcode images/masks, unknown overlap and score orientation; not detection performance or an AI-edit probability; no localization metrics computed'}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
