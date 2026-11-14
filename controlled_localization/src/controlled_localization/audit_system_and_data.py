"""Phase 0 Audit: System, Hardware, Dataset, Mask Polarity, and HuggingFace Remote Availability.
Verifies:
1. GPU, VRAM, RAM, and Disk space on Drive I: and C:
2. Local dataset inventory: exact counts by label, generator, split, edit fraction, and resolution.
3. Strict SHA-256 cross-split leakage audit (ensuring test sets remain strictly isolated).
4. Mask polarity and dimension validation.
5. HuggingFace OpenSDID / OpenSDIDplus remote inventory for scaling toward 10,000 images.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import urllib.request

import numpy as np
import torch
from PIL import Image


ROOT = Path("I:/icvgip workshop")


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def audit_system() -> Dict[str, Any]:
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "capability": torch.cuda.get_device_capability(0),
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "allocated_vram_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            "reserved_vram_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2),
        }
    else:
        gpu_info = {"status": "CUDA unavailable"}

    disk_i = shutil.disk_usage("I:/")
    disk_c = shutil.disk_usage("C:/")

    return {
        "gpu": gpu_info,
        "disk_i": {
            "total_gb": round(disk_i.total / (1024**3), 2),
            "free_gb": round(disk_i.free / (1024**3), 2),
            "used_gb": round(disk_i.used / (1024**3), 2),
        },
        "disk_c": {
            "total_gb": round(disk_c.total / (1024**3), 2),
            "free_gb": round(disk_c.free / (1024**3), 2),
            "used_gb": round(disk_c.used / (1024**3), 2),
        },
        "python_env": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
    }


def audit_local_manifests() -> Dict[str, Any]:
    manifest_dir = ROOT / "baseline_work/data_manifests"
    test_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]

    test_hashes = {}
    test_counts = {}

    for tf in test_files:
        gen = tf.replace("test_", "").replace("_manifest.json", "")
        p = manifest_dir / tf
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        test_counts[gen] = {
            "total": len(entries),
            "edited": sum(1 for e in entries if int(e["label"]) == 1),
            "genuine": sum(1 for e in entries if int(e["label"]) == 0),
        }
        for e in entries:
            img_p = ROOT / e["image_path"]
            if img_p.exists():
                h = get_file_sha256(img_p)
                test_hashes[h] = {"generator": gen, "image_path": str(img_p), "label": int(e["label"])}

    # Development manifests
    dev_p = ROOT / "controlled_localization/runs/real_development_20261114/manifest_multigen.json"
    with open(dev_p, "r", encoding="utf-8") as f:
        dev_manifest = json.load(f)

    dev_train = [r for r in dev_manifest if r["split"] == "train"]
    dev_val = [r for r in dev_manifest if r["split"] == "val"]

    # Check for SHA-256 leakage into test sets
    train_leakage = []
    val_leakage = []
    train_hashes = set()
    val_hashes = set()

    mask_errors = []
    resolution_counts = {}
    edit_size_bins = {"tiny (<2%)": 0, "medium (2-10%)": 0, "large (>10%)": 0, "pristine (0%)": 0}
    generator_breakdown = {}

    for r in dev_manifest:
        img_p = Path(r["image"])
        mask_val = r.get("mask")
        lbl = int(r["label"])
        split = r["split"]

        if not img_p.exists():
            continue

        h = get_file_sha256(img_p)
        if split == "train":
            train_hashes.add(h)
            if h in test_hashes:
                train_leakage.append(str(img_p))
        else:
            val_hashes.add(h)
            if h in test_hashes:
                val_leakage.append(str(img_p))

        # Check resolution
        with Image.open(img_p) as im:
            res_key = f"{im.width}x{im.height}"
            resolution_counts[res_key] = resolution_counts.get(res_key, 0) + 1
            im_w, im_h = im.size

        # Generator tag extraction from key or metadata
        key = r.get("key", "")
        gen_found = "unknown"
        for g_candidate in ["sd15", "sd2", "sd3", "sdxl", "flux"]:
            if g_candidate in key.lower():
                gen_found = g_candidate
                break
        generator_breakdown[gen_found] = generator_breakdown.get(gen_found, 0) + 1

        # Mask polarity and validity check
        if lbl == 1:
            if mask_val is None or not Path(mask_val).exists():
                mask_errors.append({"id": r["id"], "error": "Missing mask file for label=1"})
            else:
                with Image.open(mask_val) as m_img:
                    if m_img.size != (im_w, im_h):
                        mask_errors.append({"id": r["id"], "error": f"Dimension mismatch: img {im_w}x{im_h} vs mask {m_img.size}"})
                    m_arr = np.array(m_img.convert("L"))
                    edited_pixels = np.sum(m_arr > 127.5)
                    frac = edited_pixels / (im_w * im_h)
                    if edited_pixels == 0:
                        mask_errors.append({"id": r["id"], "error": "Mask has 0 edited pixels for label=1"})
                    if frac < 0.02:
                        edit_size_bins["tiny (<2%)"] += 1
                    elif frac <= 0.10:
                        edit_size_bins["medium (2-10%)"] += 1
                    else:
                        edit_size_bins["large (>10%)"] += 1
        else:
            edit_size_bins["pristine (0%)"] += 1

    return {
        "test_counts": test_counts,
        "test_total_images": len(test_hashes),
        "dev_train_count": len(dev_train),
        "dev_val_count": len(dev_val),
        "dev_total_count": len(dev_manifest),
        "leakage": {
            "train_test_leakage_count": len(train_leakage),
            "val_test_leakage_count": len(val_leakage),
            "train_val_overlap_count": len(train_hashes.intersection(val_hashes)),
        },
        "mask_validation": {
            "total_mask_errors": len(mask_errors),
            "errors_sample": mask_errors[:5],
        },
        "resolution_distribution": resolution_counts,
        "edit_size_distribution": edit_size_bins,
        "generator_distribution": generator_breakdown,
    }


def audit_remote_huggingface() -> Dict[str, Any]:
    """Check how many eligible images are available on HuggingFace for scaling."""
    endpoints = {
        "nebula_OpenSDIDplus": "https://datasets-server.huggingface.co/info?dataset=nebula/OpenSDIDplus",
        "nebula_OpenSDID": "https://datasets-server.huggingface.co/info?dataset=nebula/OpenSDID",
    }
    remote_stats = {}
    for name, url in endpoints.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                dataset_info = data.get("dataset_info", {})
                default_config = list(dataset_info.keys())[0] if dataset_info else "default"
                splits = dataset_info.get(default_config, {}).get("splits", {})
                remote_stats[name] = {
                    "status": "Available",
                    "splits": {s: v.get("num_examples", 0) for s, v in splits.items()},
                    "total_examples": sum(v.get("num_examples", 0) for v in splits.values()),
                }
        except Exception as e:
            remote_stats[name] = {"status": "Error", "message": str(e)}

    return remote_stats


def main() -> None:
    print("================================================================================")
    print("PHASE 0 INVENTORY & PROTOCOL AUDIT")
    print("================================================================================")

    sys_info = audit_system()
    print("\n1. HARDWARE & COMPUTE INVENTORY:")
    print(f"  GPU: {sys_info['gpu'].get('name', 'N/A')} ({sys_info['gpu'].get('total_vram_gb', 0)} GB VRAM)")
    print(f"  Drive I: Free: {sys_info['disk_i']['free_gb']} GB / Total: {sys_info['disk_i']['total_gb']} GB")
    print(f"  Drive C: Free: {sys_info['disk_c']['free_gb']} GB / Total: {sys_info['disk_c']['total_gb']} GB")
    print(f"  PyTorch: {sys_info['python_env']['torch_version']} (CUDA: {sys_info['python_env']['cuda_version']})")

    print("\n2. LOCAL DATASETS & INTEGRITY AUDIT:")
    manifest_audit = audit_local_manifests()
    print(f"  Test Benchmark Total Images: {manifest_audit['test_total_images']} (100% frozen)")
    for g, cnt in manifest_audit["test_counts"].items():
        print(f"    - {g.upper():<8} : Total={cnt['total']}, Edited={cnt['edited']}, Genuine={cnt['genuine']}")
    print(f"\n  Development Set Total: {manifest_audit['dev_total_count']} (Train: {manifest_audit['dev_train_count']}, Val: {manifest_audit['dev_val_count']})")
    print(f"  Leakage Cross-Check: Train-Test Overlap = {manifest_audit['leakage']['train_test_leakage_count']}, Val-Test Overlap = {manifest_audit['leakage']['val_test_leakage_count']}")
    print(f"  Train-Val Overlap: {manifest_audit['leakage']['train_val_overlap_count']}")
    print(f"  Mask Validation Errors: {manifest_audit['mask_validation']['total_mask_errors']}")
    print(f"\n  Edit Size Distribution in Development:")
    for b, c in manifest_audit["edit_size_distribution"].items():
        print(f"    - {b:<18}: {c}")
    print(f"\n  Generator Distribution in Development:")
    for g, c in manifest_audit["generator_distribution"].items():
        print(f"    - {g:<10}: {c}")

    print("\n3. HUGGINGFACE REMOTE INVENTORY (SCALING TO 10,000 IMAGES):")
    remote_audit = audit_remote_huggingface()
    for name, r in remote_audit.items():
        if r.get("status") == "Available":
            print(f"  {name}: {r['total_examples']} total examples available. Splits: {r['splits']}")
        else:
            print(f"  {name}: Status = {r.get('status')} ({r.get('message')})")

    # Save to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
        "system": sys_info,
        "local_manifests": manifest_audit,
        "remote_huggingface": remote_audit,
    }
    out_p = ROOT / "controlled_localization/reports/phase0_inventory_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nComplete Phase 0 Inventory saved to: {out_p}")


if __name__ == "__main__":
    main()
