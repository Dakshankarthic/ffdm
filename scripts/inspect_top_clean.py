import json
from pathlib import Path

with open("scripts/clean_candidates.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total entries: {len(data)}")
for i, d in enumerate(data[:15]):
    sid = d["sample_id"]
    s = d["s_hybrid"]
    mp = d["max_prob"]
    fa = d["fa_pixels"]
    w, h = d["w"], d["h"]
    p = d["image_path"]
    print(f"[{i:2d}] {sid} | s={s:.4f} | max_P={mp:.4f} | FA={fa} | {w}x{h} | {p}")
