"""Strict on-disk four-view loader; never substitutes synthetic examples."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .schema import ManifestRecord

def load_research_inputs(manifest_path, schedule_path, validation_path):
    def read(path):
        p = Path(path)
        records = [ManifestRecord(**json.loads(line)) for line in p.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        for r in records:
            r.validate()
            if r.status != 'verified' or not r.mask_verified or not r.geometry_verified:
                raise ValueError(f'Unverified research input: {r.sample_id}')
            for key in ('image_path', 'mask_path', 'ignore_mask_path'):
                value = getattr(r, key)
                if value:
                    q = Path(value)
                    q = q if q.is_absolute() else p.parent / q
                    if not q.is_file(): raise FileNotFoundError(q)
                    setattr(r, key, str(q.resolve()))
        return records
    train, val = read(manifest_path), read(validation_path)
    if not train or not val: raise ValueError('Nonempty train and validation manifests required')
    if any(r.split != 'train' for r in train) or any(r.split != 'val' for r in val):
        raise ValueError('Incorrect split labels')
    if {r.source_group_id for r in train} & {r.source_group_id for r in val}:
        raise ValueError('Training/validation source overlap')
    if {r.image_sha256 for r in train} & {r.image_sha256 for r in val}:
        raise ValueError('Training/validation image hash overlap')
    by_group = {}
    for r in train:
        roles = by_group.setdefault(r.pair_group_id, {})
        if r.view_role in roles: raise ValueError('Duplicate group/view role')
        roles[r.view_role] = r
    schedule = [json.loads(s) for s in Path(schedule_path).read_text().splitlines() if s.strip()]
    if not schedule: raise ValueError('Empty schedule')
    groups = []
    for item in schedule:
        roles = by_group[item['pair_group_id']]
        if set(roles) != set('ABCD'): raise ValueError('Four views required')
        x, y, size = int(item['x']), int(item['y']), int(item.get('size',384))
        if x < 0 or y < 0 or size != 384: raise ValueError('Invalid crop')
        g = {}
        shapes = []
        for role, r in roles.items():
            with Image.open(r.image_path) as im: a = np.array(im.convert('RGB'),dtype=np.float32)/255
            shapes.append(a.shape)
            if y+size>a.shape[0] or x+size>a.shape[1]: raise ValueError('Crop outside image')
            g['img_'+role.lower()] = torch.from_numpy(a[y:y+size,x:x+size].copy()).permute(2,0,1)[None]
        if len(set(shapes)) != 1: raise ValueError('View geometry mismatch')
        with Image.open(roles['C'].mask_path) as im: mask=np.asarray(im.convert('L'))
        if mask.shape != shapes[0][:2] or not set(np.unique(mask)).issubset({0,255}):
            raise ValueError('Requires aligned lossless binary mask')
        mask=mask.astype(np.float32)/255
        g['mask_c']=torch.from_numpy(mask[y:y+size,x:x+size].copy())[None,None]
        if any(r.ignore_mask_path for r in roles.values()):
            raise ValueError('Ignore-mask loading not yet supported; cannot silently ignore it')
        g['valid_mask']=torch.ones_like(g['mask_c'])
        g['outside_eligible']=False
        groups.append(g)
    return groups, val
