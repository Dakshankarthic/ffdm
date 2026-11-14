import requests
import io
import numpy as np
from PIL import Image

def inspect_mask(dataset, split, offset):
    p = {'dataset': dataset, 'config': 'default', 'split': split, 'offset': offset, 'length': 1}
    r = requests.get('https://datasets-server.huggingface.co/rows', params=p, timeout=20)
    if r.status_code != 200:
        print(f"Failed to fetch {split} {offset}: {r.status_code}")
        return
    row = r.json()['rows'][0]['row']
    mask_info = row.get('mask')
    img_info = row.get('image')
    key = row.get('key')
    print(f"\n--- Checking key: {key} (split: {split}, offset: {offset}) ---")
    if not mask_info or not mask_info.get('src'):
        print("No mask found!")
        return
    
    r_img = requests.get(img_info['src'], timeout=20)
    r_mask = requests.get(mask_info['src'], timeout=20)
    
    img = Image.open(io.BytesIO(r_img.content))
    mask = Image.open(io.BytesIO(r_mask.content))
    
    mask_np = np.array(mask)
    print(f"Image: size={img.size}, format={img.format}, mode={img.mode}")
    print(f"Mask: size={mask.size}, format={mask.format}, mode={mask.mode}, dtype={mask_np.dtype}")
    print(f"Mask values: min={mask_np.min()}, max={mask_np.max()}, unique={np.unique(mask_np)[:10]}")
    
    # Check threshold > 127
    binary = (mask_np > 127)
    pct = binary.sum() / binary.size * 100
    print(f"Edit percentage (>127): {pct:.2f}% ({binary.sum()} / {binary.size} pixels)")

inspect_mask('nebula/OpenSDI_test', 'sd15', 10000)
inspect_mask('nebula/OpenSDI_test', 'sd15', 10005)
inspect_mask('nebula/OpenSDI_test', 'sd2', 10000)
inspect_mask('nebula/OpenSDI_test', 'flux', 10000)

