import os
import io
import json
import csv
import hashlib
import time
import requests
from pathlib import Path
from PIL import Image
import numpy as np

DATA_ROOT = Path("I:/icvgip workshop/baseline_work/data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
MANIFESTS_ROOT = Path("I:/icvgip workshop/baseline_work/data_manifests")
MANIFESTS_ROOT.mkdir(parents=True, exist_ok=True)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def classify_tier(pct: float) -> str:
    if pct < 2.0:
        return "<2%"
    elif pct <= 10.0:
        return "2-10%"
    else:
        return ">10%"

def get_with_retry(url, params=None, max_retries=6, backoff=3.0):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=45)
            if r.status_code == 200:
                return r
            elif r.status_code in [500, 502, 503, 504, 429]:
                print(f"  [Attempt {attempt}/{max_retries}] HTTP {r.status_code} for {url}, backing off {backoff*attempt:.1f}s...", flush=True)
                time.sleep(backoff * attempt)
            else:
                r.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"  [Attempt {attempt}/{max_retries}] Network error: {e}, backing off {backoff*attempt:.1f}s...", flush=True)
            time.sleep(backoff * attempt)
    # Final try
    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    return r

def get_rows_bulk(dataset, split, offset, length=10):
    p = {'dataset': dataset, 'config': 'default', 'split': split, 'offset': offset, 'length': length}
    r = get_with_retry('https://datasets-server.huggingface.co/rows', params=p, max_retries=6, backoff=3.0)
    return r.json().get('rows', [])

def process_and_save_sample(row_obj, dataset, split, target_folder):
    row = row_obj['row']
    row_idx = row_obj['row_idx']
    key = row['key']
    label = row['label']
    img_info = row['image']
    mask_info = row.get('mask')
    clean_key = key.replace("/", "_").replace("\\", "_")
    
    # Download image
    r_img = get_with_retry(img_info['src'])
    img_bytes = r_img.content
    img_sha = sha256_bytes(img_bytes)
    img_pil = Image.open(io.BytesIO(img_bytes))
    img_pil.verify()
    img_w, img_h = img_pil.size
    img_fmt = img_pil.format or "JPEG"
    
    img_filename = f"{clean_key}_img.{img_fmt.lower()}"
    img_path = target_folder / img_filename
    img_path.write_bytes(img_bytes)
    
    mask_path_str = ""
    mask_sha = ""
    mask_fmt = ""
    edit_area_pct = 0.0
    edit_tier = "none"
    edit_type = "genuine"
    
    if label == 1:
        edit_type = "partial_inpainting"
        if not mask_info or not mask_info.get('src'):
            raise RuntimeError(f"Fake sample {key} has no mask!")
        r_mask = get_with_retry(mask_info['src'])
        mask_bytes = r_mask.content
        mask_sha = sha256_bytes(mask_bytes)
        mask_pil = Image.open(io.BytesIO(mask_bytes))
        mask_pil.verify()
        mask_w, mask_h = mask_pil.size
        assert (mask_w, mask_h) == (img_w, img_h), f"Dim mismatch: mask {(mask_w, mask_h)} vs img {(img_w, img_h)}"
        mask_fmt = mask_pil.format or "JPEG"
        mask_filename = f"{clean_key}_mask.{mask_fmt.lower()}"
        mask_target_path = target_folder / mask_filename
        mask_target_path.write_bytes(mask_bytes)
        mask_path_str = str(mask_target_path.relative_to(Path("I:/icvgip workshop")))
        
        m_arr = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
        bin_mask = (m_arr > 127)
        edit_area_pct = float(bin_mask.sum() / bin_mask.size * 100.0)
        edit_tier = classify_tier(edit_area_pct)
        
    entry = {
        "sample_id": f"{split}_{row_idx:06d}_{clean_key}",
        "dataset": dataset,
        "dataset_revision": "main",
        "source_id": key.split("/")[-1].split(".")[0],
        "split": split,
        "generator": split if split != "default" else "sd15",
        "edit_type": edit_type,
        "image_path": str(img_path.relative_to(Path("I:/icvgip workshop"))),
        "mask_path": mask_path_str,
        "label": int(label),
        "image_sha256": img_sha,
        "mask_sha256": mask_sha,
        "width": img_w,
        "height": img_h,
        "image_format": img_fmt,
        "mask_format": mask_fmt,
        "edit_area_pct": round(edit_area_pct, 4),
        "edit_tier": edit_tier,
        "source_url": f"https://huggingface.co/datasets/{dataset}",
        "license": "CC BY-SA 4.0",
        "notes": f"Row index {row_idx} in {dataset}/{split}"
    }
    return entry

def build_split_bulk(name, dataset, split, fake_offset=10000, real_offset=15000, n_each=10):
    target_folder = DATA_ROOT / name
    target_folder.mkdir(parents=True, exist_ok=True)
    manifest_json = MANIFESTS_ROOT / f"{name}_manifest.json"
    manifest_csv = MANIFESTS_ROOT / f"{name}_manifest.csv"
    
    if manifest_json.exists():
        print(f"Split {name} manifest already exists, loading...", flush=True)
        with open(manifest_json, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"\n=======================================================", flush=True)
    print(f"Building split: {name} ({n_each} fake from {fake_offset}, {n_each} real from {real_offset})", flush=True)
    print(f"=======================================================", flush=True)
    
    # 1. Fetch fake rows
    print(f"Fetching {n_each} fake rows from {dataset} {split} offset {fake_offset}...", flush=True)
    fake_rows = get_rows_bulk(dataset, split, fake_offset, length=n_each)
    time.sleep(1.0)
    
    # 2. Fetch real rows
    print(f"Fetching {n_each} real rows from {dataset} {split} offset {real_offset}...", flush=True)
    real_rows = get_rows_bulk(dataset, split, real_offset, length=n_each)
    
    entries = []
    for r in fake_rows:
        e = process_and_save_sample(r, dataset, split, target_folder)
        entries.append(e)
        print(f"  Processed Fake: {e['sample_id']} (tier: {e['edit_tier']}, pct: {e['edit_area_pct']}%)", flush=True)
        
    for r in real_rows:
        e = process_and_save_sample(r, dataset, split, target_folder)
        entries.append(e)
        print(f"  Processed Real: {e['sample_id']} (label: {e['label']})", flush=True)
        
    entries.sort(key=lambda x: x["sample_id"])
    
    manifest_data = {
        "split_name": name,
        "dataset": dataset,
        "split": split,
        "num_samples": len(entries),
        "num_real": sum(1 for e in entries if e["label"] == 0),
        "num_fake": sum(1 for e in entries if e["label"] == 1),
        "tiers": {
            "<2%": sum(1 for e in entries if e["edit_tier"] == "<2%"),
            "2-10%": sum(1 for e in entries if e["edit_tier"] == "2-10%"),
            ">10%": sum(1 for e in entries if e["edit_tier"] == ">10%"),
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": entries
    }
    
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
        writer.writeheader()
        writer.writerows(entries)
        
    print(f"Saved {name} manifest with {len(entries)} items (Real: {manifest_data['num_real']}, Fake: {manifest_data['num_fake']})", flush=True)
    print(f"Tiers: {manifest_data['tiers']}", flush=True)
    return manifest_data

if __name__ == "__main__":
    # Validation split val_sd15
    val_json = MANIFESTS_ROOT / "val_sd15_manifest.json"
    if val_json.exists():
        with open(val_json, "r", encoding="utf-8") as f:
            val_manifest = json.load(f)
        print(f"Loaded existing val_sd15 manifest ({val_manifest['num_samples']} items).", flush=True)
    else:
        val_manifest = build_split_bulk("val_sd15", "nebula/OpenSDI_train", "sd15", fake_offset=1000, real_offset=100000, n_each=10)
        
    # Build test splits from OpenSDI_test (sd15, sd2, sdxl, sd3, flux)
    test_generators = ["sd15", "sd2", "sdxl", "sd3", "flux"]
    all_test_manifests = {}
    for gen in test_generators:
        all_test_manifests[gen] = build_split_bulk(f"test_{gen}", "nebula/OpenSDI_test", gen, fake_offset=10000, real_offset=15000, n_each=10)
        time.sleep(1.0)
        
    # Audit split integrity: cross-split SHA256 check
    print("\n=======================================================", flush=True)
    print("Auditing cross-split image hash overlap...", flush=True)
    print("=======================================================", flush=True)
    all_splits = [val_manifest] + list(all_test_manifests.values())
    hash_to_sample = {}
    overlap_found = False
    for m in all_splits:
        split_name = m["split_name"]
        for e in m["entries"]:
            h = e["image_sha256"]
            if h in hash_to_sample:
                prev = hash_to_sample[h]
                print(f"LEAKAGE DETECTED: image SHA256 {h} in {split_name} ({e['sample_id']}) matches {prev['split']} ({prev['sample_id']})!", flush=True)
                overlap_found = True
            else:
                hash_to_sample[h] = {"split": split_name, "sample_id": e["sample_id"]}
                
    if not overlap_found:
        print("AUDIT PASSED: ZERO image SHA256 overlap across validation and all test splits!", flush=True)
    else:
        raise RuntimeError("Integrity audit failed: duplicate image hashes detected across splits!")
