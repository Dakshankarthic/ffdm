from pathlib import Path
import hashlib,json,requests,concurrent.futures
from PIL import Image
from io import BytesIO
base=Path('research_data/opensdi_development_preview');base.mkdir(parents=True,exist_ok=True)
offsets=[1000,11000,21000,31000,41000,51000,61000,71000,100000,110000,120000,130000,140000,150000,160000,170000]
def get(offset):
 p={'dataset':'nebula/OpenSDI_train','config':'default','split':'sd15','offset':offset,'length':1}
 r=requests.get('https://datasets-server.huggingface.co/rows',params=p,timeout=60);r.raise_for_status(); row=r.json()['rows'][0]['row'];d={'row_index':offset,'key':row['key'],'label':row['label'],'assets':{}}
 for k in ['image','mask']:
  a=row.get(k)
  if not a or not a.get('src'):continue
  z=requests.get(a['src'],timeout=60);z.raise_for_status(); im=Image.open(BytesIO(z.content));im.verify(); ext={'JPEG':'jpg','PNG':'png','WEBP':'webp'}.get(im.format,'bin'); name=f'{offset:06}_{k}.{ext}';(base/name).write_bytes(z.content);d['assets'][k]={'file':name,'sha256':hashlib.sha256(z.content).hexdigest(),'size':[im.width,im.height],'format':im.format}
 return d
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex: items=list(ex.map(get,offsets))
manifest={'dataset':'nebula/OpenSDI_train','split':'sd15','source':'https://huggingface.co/datasets/nebula/OpenSDI_train','selection':'16 systematically spaced training viewer rows, selected for quick diagnostics, not random, not independent test; source grouping unchecked','note':'Viewer cached assets may be transcoded; JPEG masks require polarity and label-semantic validation before localization metrics; dataset CC BY-SA 4.0 and academic-use notice','items':items}
(base/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('downloaded',len(items),'labels',{k:sum(v['label']==k for v in items) for k in (0,1)},'partial',sum(v['key'].startswith('partial/') for v in items),flush=True)
