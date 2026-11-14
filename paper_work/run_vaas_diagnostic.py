import os,json,time
from pathlib import Path
os.environ['HF_HOME']=str(Path('baselines/model_cache').resolve())
import torch, numpy as np
from vaas.inference.pipeline import VAASPipeline
p=Path('research_data/opensdi_viewer_preview/row_0000_image.jpg').resolve()
out=Path('paper_work/diagnostic_01');out.mkdir(parents=True,exist_ok=True)
print('torch:',torch.__version__,'cuda:',torch.cuda.is_available(),flush=True)
model=VAASPipeline.from_pretrained('OBA-Research/vaas',device='cuda' if torch.cuda.is_available() else 'cpu')
print('model loaded',flush=True)
r=model(str(p)); arr=np.asarray(r.pop('anomaly_map'));np.save(out/'anomaly_map.npy',arr)
report={'image':str(p),'source':'OpenSDI_train sd15 row 0; diagnostic only','model':'OBA-Research/vaas','torch':torch.__version__,'device':'cuda' if torch.cuda.is_available() else 'cpu','scores':r,'anomaly_map_shape':list(arr.shape),'warning':'Model outputs are not validated AI-edit probabilities; no performance conclusion.'}
(out/'result.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
print(json.dumps(report,indent=2,default=str),flush=True)
