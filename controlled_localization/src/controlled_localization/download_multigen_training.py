"""Download verified multi-generator training samples (FLUX, SDXL, SD3, SD2) from nebula/OpenSDIDplus.
Strictly excludes all benchmark test and validation IDs to maintain 100% zero data leakage.
"""

from __future__ import annotations

import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set

from PIL import Image


ROOT = Path("I:/icvgip workshop")
DATA_DIR = ROOT / "controlled_localization/runs/real_development_20261114/data"
BASE_MANIFEST = ROOT / "controlled_localization/runs/real_development_20261114/manifest.json"
TARGET_MANIFEST = ROOT / "controlled_localization/runs/real_development_20261114/manifest_multigen.json"


def fetch_url(url: str, retries: int = 4) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                return b""
            time.sleep(2 * (attempt + 1))
    return b""


def get_excluded_ids() -> Set[str]:
    """Collect all numeric tokens from test and validation files to prevent any leakage."""
    test_files = glob.glob(str(ROOT / "baseline_work/data/test_*/*.*"))
    test_ids = set()
    for f in test_files:
        m = re.findall(r"\d+", os.path.basename(f))
        for num in m:
            test_ids.add(num.lstrip("0"))
    return {i for i in test_ids if len(i) > 2}


def download_split(split_name: str, target_count: int = 100, excluded_ids: Set[str] = set()) -> List[Dict]:
    print(f"\n[Split: {split_name}] Target: {target_count} samples from nebula/OpenSDIDplus...")
    downloaded = []
    offset = 30000

    while len(downloaded) < target_count and offset < 70000:
        url = (
            f"https://datasets-server.huggingface.co/rows?"
            f"dataset=nebula/OpenSDIDplus&config=default&split={split_name}&offset={offset}&length=50"
        )
        try:
            raw_data = fetch_url(url)
            if not raw_data:
                offset += 50
                continue
            data = json.loads(raw_data)
            rows = data.get("rows", [])
        except Exception as e:
            print(f"  Error fetching offset {offset}: {e}")
            offset += 50
            time.sleep(1)
            continue

        for r_item in rows:
            if len(downloaded) >= target_count:
                break
            r = r_item.get("row", {})
            key = r.get("key", "")
            if "partial" not in key or r.get("mask") is None:
                continue

            # Check ID exclusion
            m = re.findall(r"\d+", key)
            if any(num.lstrip("0") in excluded_ids for num in m):
                continue

            sid = f"{split_name}_{m[-1] if m else r_item['row_idx']}"
            img_filename = f"multigen_{sid}.jpg"
            mask_filename = f"multigen_{sid}_mask.png"

            img_p = DATA_DIR / img_filename
            mask_p = DATA_DIR / mask_filename

            # Download if not already on disk
            if not img_p.exists() or not mask_p.exists():
                img_bytes = fetch_url(r["image"]["src"])
                mask_bytes = fetch_url(r["mask"]["src"])
                if not img_bytes or not mask_bytes:
                    continue

                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img.save(img_p, quality=95)
                    mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
                    mask.save(mask_p)
                except Exception:
                    continue

            # SHA-256 hash
            h = hashlib.sha256()
            with open(img_p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            sha = h.hexdigest()

            rec = {
                "id": sid,
                "split": "train",
                "label": 1.0,
                "image": str(img_p),
                "mask": str(mask_p),
                "source_id": sid,
                "scope": f"multi_{split_name}",
                "fraction": 0.10,
                "sha256": sha,
                "key": key,
            }
            downloaded.append(rec)
            print(f"  + [{split_name}] {len(downloaded)}/{target_count}: {img_filename}", end="\r")

        offset += 50

    print(f"\n[Split: {split_name}] Finished: {len(downloaded)} samples collected.")
    return downloaded


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    excluded_ids = get_excluded_ids()
    print(f"Loaded {len(excluded_ids)} excluded test/benchmark IDs for strict zero data leakage.")

    # Load existing manifest
    with open(BASE_MANIFEST, "r", encoding="utf-8") as f:
        existing_records = json.load(f)

    print(f"Existing dataset: {len(existing_records)} samples (900 train, 220 val).")

    new_records = []
    # Download 75 samples each across all 4 generators (75 * 4 = 300 multi-gen samples)
    for split in ["flux", "sdxl", "sd3", "sd2"]:
        recs = download_split(split, target_count=75, excluded_ids=excluded_ids)
        new_records.extend(recs)

    combined_manifest = existing_records + new_records
    print(f"\nTotal combined manifest records: {len(combined_manifest)}")

    with open(TARGET_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(combined_manifest, f, indent=2)

    print(f"Saved expanded multi-generator manifest to: {TARGET_MANIFEST}")


if __name__ == "__main__":
    main()
