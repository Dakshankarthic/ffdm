"""Evaluate only the explicitly supplied new checkpoint, with validation calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from controlled_localization.eval_score_distributions import evaluate_checkpoint

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    records = json.loads(Path(a.manifest).read_text())
    checkpoint = Path(a.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    result = evaluate_checkpoint(checkpoint, 'super_v2',
        [r for r in records if r['split']=='val'],
        [r for r in records if r['split']=='test'], torch.device('cuda'))
    result['checkpoint_absolute_path'] = str(checkpoint)
    result['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result['scope'] = 'User-provided held-out split; previously inspected examples remain development data'
    Path(a.output).write_text(json.dumps({'kaggle_trained_model':result}, indent=2))
