"""Validate an attached dataset and translate portable or legacy manifest paths."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image

def prepare(manifest, data_root, output):
    root = Path(data_root).resolve()
    records = json.loads(Path(manifest).read_text(encoding='utf-8-sig'))
    if not isinstance(records, list):
        raise ValueError('Manifest must be a list of image records')
    def resolve(value):
        if not value:
            raise ValueError('Missing required file path')
        raw = str(value).replace('\\', '/')
        candidates = [Path(raw), root / raw]
        marker = 'icvgip workshop/'
        if marker in raw:
            candidates.append(root / raw.split(marker, 1)[1])
        for p in candidates:
            if p.is_file():
                return p.resolve()
        raise FileNotFoundError(f'File not found: {value}. Attach the image/mask dataset and set DATA_ROOT.')
    sources, hashes, counts = {}, {}, {}
    for r in records:
        split = r.get('split')
        if split not in ('train', 'val', 'test') or r.get('label') not in (0, 1):
            raise ValueError('Each record requires split=train/val/test and label=0/1')
        p = resolve(r.get('image', r.get('image_path')))
        r['image'] = r['image_path'] = str(p)
        with Image.open(p) as im:
            im.load()
            size = im.size
        source = str(r.get('source_id') or '').strip()
        if not source:
            raise ValueError(f'Missing source_id for {p}; group original and edited derivatives before splitting')
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        for mapping, key in ((sources, source), (hashes, digest)):
            if key in mapping and mapping[key] != split:
                raise ValueError(f'Cross-split source or exact-image overlap: {p}')
            mapping[key] = split
        r['sha256'] = digest
        if r['label'] == 1:
            m = resolve(r.get('mask'))
            with Image.open(m) as im:
                im = im.convert('L')
                if im.size != size or im.point(lambda v: 255 if v > 127 else 0).getbbox() is None:
                    raise ValueError(f'Mask must align and contain white edited pixels: {m}')
            r['mask'] = str(m)
        counts.setdefault(split, [0, 0])[int(r['label'])] += 1
    for split in ('train', 'val', 'test'):
        if split not in counts or min(counts[split]) == 0:
            raise ValueError(f'{split} needs genuine and edited examples')
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Validated counts [genuine, edited]:', counts)
    print('Source IDs and exact hashes checked; near-duplicate provenance still requires dataset curation.')
    return records

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    prepare(a.manifest, a.data_root, a.output)
