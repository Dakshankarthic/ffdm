"""End-to-end dataset expansion, scale training, benchmark evaluation, and email notification."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from controlled_localization.models.detector import ControlledLocalizationDetector
from controlled_localization.losses.segmentation import compute_supervised_segmentation_loss
from controlled_localization.losses.consistency import compute_directed_consistency_loss
from controlled_localization.data.canonicalize import apply_recipe
from controlled_localization.infer import run_single_image_inference
from controlled_localization.policy import InferencePolicy
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
BASE = ROOT / "controlled_localization/runs/real_development_20261114"
DATA_DIR = BASE / "data"


def fetch_url(url: str, retries: int = 4) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(2 * (attempt + 1))
    return b""


def expand_dataset(target_total: int = 800) -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_p = BASE / "manifest.json"
    existing_records = []
    seen_ids = set()
    seen_hashes = set()

    if manifest_p.exists():
        existing_records = json.loads(manifest_p.read_text(encoding="utf-8"))
        for r in existing_records:
            seen_ids.add(r["source_id"])
            seen_hashes.add(r["sha256"])

    print(f"Existing dataset has {len(existing_records)} samples.")
    if len(existing_records) >= target_total:
        return existing_records

    # Collect excluded test/pilot IDs
    excluded_ids = set(seen_ids)
    for p in (ROOT / "baseline_work/data_manifests").glob("*.json"):
        for ent in json.loads(p.read_text(encoding="utf-8")).get("entries", []):
            excluded_ids.add(str(ent.get("source_id", "")))

    needed = target_total - len(existing_records)
    print(f"Downloading {needed} additional diverse samples from OpenSDI_train...")

    # Query varied offsets across the 200k train split
    offsets = [12000, 16000, 24000, 32000, 48000, 64000, 80000, 96000, 112000, 128000, 144000, 160000]
    candidate_rows = []

    for off in offsets:
        if len(candidate_rows) >= needed * 2:
            break
        try:
            q = urllib.parse.urlencode({"dataset": "nebula/OpenSDI_train", "config": "default", "split": "sd15", "offset": off, "length": 80})
            data = json.loads(fetch_url("https://datasets-server.huggingface.co/rows?" + q))
            candidate_rows.extend(data.get("rows", []))
            print(f"Fetched metadata offset {off} (+{len(data.get('rows', []))} candidates)...")
        except Exception as err:
            print(f"Offset {off} error: {err}")

    def download_sample(item: dict) -> dict | None:
        try:
            r = item["row"]
            sid = Path(r["key"]).stem
            if sid in excluded_ids or ("partial" not in r["key"]):
                return None

            label = int(r["label"])
            name = f"exp_{item['row_idx']}"
            img_bytes = fetch_url(r["image"]["src"])
            img_hash = hashlib.sha256(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                return None

            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_file = DATA_DIR / f"{name}.jpg"
            img_file.write_bytes(img_bytes)

            mask_file = None
            mask_arr = np.zeros((im.height, im.width), dtype=np.uint8)

            if label == 1:
                if not r.get("mask"):
                    return None
                m_bytes = fetch_url(r["mask"]["src"])
                m_img = Image.open(io.BytesIO(m_bytes)).convert("RGB")
                if m_img.size != im.size:
                    return None
                mask_arr = (np.asarray(m_img).mean(axis=2) > 127.5).astype(np.uint8)
                if mask_arr.sum() == 0:
                    return None
                mask_file = DATA_DIR / f"{name}_mask.png"
                Image.fromarray(mask_arr * 255).save(mask_file)

            return {
                "id": name,
                "source_id": sid,
                "label": label,
                "image": str(img_file),
                "mask": str(mask_file) if mask_file else None,
                "sha256": img_hash,
                "fraction": float(mask_arr.mean()),
                "key": r["key"],
                "scope": "OpenSDI_train scale dataset",
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [res for res in pool.map(download_sample, candidate_rows) if res is not None]

    print(f"Downloaded {len(results)} new valid samples.")
    combined = existing_records + results

    # Partition source-disjoint train/val
    for rec in combined:
        if "split" not in rec:
            h = int(hashlib.sha256(rec["source_id"].encode()).hexdigest()[:8], 16)
            rec["split"] = "val" if h % 5 == 0 else "train"

    manifest_p.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Total dataset ready: {len(combined)} samples ({sum(r['split']=='train' for r in combined)} train, {sum(r['split']=='val' for r in combined)} val).")
    return combined


def load_img_and_mask(r: dict) -> tuple[Image.Image, np.ndarray]:
    im = Image.open(r["image"]).convert("RGB")
    if not r.get("mask"):
        m = np.zeros((im.height, im.width), dtype=np.float32)
    else:
        m = np.asarray(Image.open(r["mask"]), dtype=np.float32) / 255.0
    return im, m


def main() -> None:
    seed_all(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Expand dataset
    records = expand_dataset(target_total=700)
    train_rows = [r for r in records if r["split"] == "train"]
    val_rows = [r for r in records if r["split"] == "val"]

    out_dir = ROOT / "controlled_localization/runs/scale_multigen_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Model setup
    init_ckpt = ROOT / "controlled_localization/runs/high_accuracy_real_t2/best.pt"
    if not init_ckpt.exists():
        init_ckpt = ROOT / "controlled_localization/runs/real_development_20261114/T1_extended/best.pt"

    print(f"Loading initial model weights from: {init_ckpt}")
    model = ControlledLocalizationDetector(pretrained_rgb=False)
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    st = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(st)
    model.to(device)
    model.train()

    # Pre-cache training data
    print("Pre-caching images into memory...")
    cache = {r["id"]: load_img_and_mask(r) for r in train_rows}
    classes = {
        0: [r for r in train_rows if r["label"] == 0],
        1: [r for r in train_rows if r["label"] == 1],
    }

    total_steps = 1000
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.rgb_encoder.parameters() if p.requires_grad], "lr": 6e-5},
            {"params": list(model.res_encoder.parameters()) + list(model.decoder.parameters()), "lr": 2e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    recipes = ["identity", "jpeg_q70", "jpeg_q85", "jpeg_q95", "blur_s05"]
    rng = np.random.default_rng(42)

    best_iou = -1.0
    start_time = time.perf_counter()
    print(f"\nStarting scale training for {total_steps} steps...")

    for step in range(1, total_steps + 1):
        xs, ys, is_clean = [], [], []

        for label in [0, 1]:
            pool = classes[label]
            r = pool[int(rng.integers(len(pool)))]
            im, m = cache[r["id"]]

            recipe = recipes[int(rng.integers(len(recipes)))]
            processed, _ = apply_recipe(im, recipe)

            a = np.asarray(im, dtype=np.float32) / 255.0
            b = np.asarray(processed, dtype=np.float32) / 255.0

            h, w = m.shape
            ph, pw = max(0, 384 - h), max(0, 384 - w)
            if ph > 0 or pw > 0:
                a = np.pad(a, ((0, ph), (0, pw), (0, 0)), mode="edge")
                b = np.pad(b, ((0, ph), (0, pw), (0, 0)), mode="edge")
                m = np.pad(m, ((0, ph), (0, pw)))
            h, w = m.shape

            if label == 1 and rng.random() < 0.70 and (m > 0.5).any():
                pts = np.argwhere(m > 0.5)
                cy, cx = pts[int(rng.integers(len(pts)))]
                x = int(np.clip(cx - 192, 0, w - 384))
                y = int(np.clip(cy - 192, 0, h - 384))
            else:
                x = int(rng.integers(max(1, w - 384 + 1)))
                y = int(rng.integers(max(1, h - 384 + 1)))

            xs.extend([a[y : y + 384, x : x + 384].transpose(2, 0, 1), b[y : y + 384, x : x + 384].transpose(2, 0, 1)])
            ys.extend([m[y : y + 384, x : x + 384][None], m[y : y + 384, x : x + 384][None]])
            is_clean.extend([label == 0, label == 0])

        x_t = torch.tensor(np.stack(xs), device=device)
        y_t = torch.tensor(np.stack(ys), device=device)
        clean_mask = torch.tensor(is_clean, device=device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            z = model(x_t)["mask_logits"]
            l_seg, _, _ = compute_supervised_segmentation_loss(z, y_t)
            
            # Clean false-alarm penalty
            z_clean = z[clean_mask]
            l_clean = F.softplus(z_clean).mean() + 1.5 * torch.clamp(torch.sigmoid(z_clean) - 0.05, min=0.0).mean()
            
            # Directed consistency
            l_cons = compute_directed_consistency_loss(z[1::2], z[0::2], y_t[0::2])
            
            loss = l_seg + 1.2 * l_clean + 0.15 * l_cons

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % 50 == 0:
            elapsed = int(time.perf_counter() - start_time)
            print(f"Step {step:4d}/{total_steps} | Loss: {loss:.4f} (seg: {l_seg:.4f}, clean: {l_clean:.4f}, cons: {l_cons:.4f}) | {elapsed}s")

        if step % 100 == 0 or step == total_steps:
            # Validation check
            model.eval()
            policy = InferencePolicy(policy_name="val", t_mask=0.50, t_image=0.50)
            val_ious = []
            for r in val_rows[:30]:
                if r["label"] == 1:
                    res = run_single_image_inference(r["image"], model, policy, device)
                    pred = res["merged_prob"] > 0.50
                    _, mask_arr = load_img_and_mask(r)
                    gt = mask_arr > 0.50
                    inter = (pred & gt).sum()
                    union = (pred | gt).sum()
                    if union > 0:
                        val_ious.append(inter / union)

            cur_iou = float(np.mean(val_ious)) if val_ious else 0.0
            print(f"--> [Step {step}] Validation Edited IoU: {cur_iou:.4f}")

            torch.save({"step": step, "model_state_dict": model.state_dict()}, out_dir / "latest.pt")
            if cur_iou > best_iou:
                best_iou = cur_iou
                print(f"*** New BEST model saved at step {step} (IoU: {best_iou:.4f})! ***")
                torch.save({"step": step, "model_state_dict": model.state_dict(), "val_iou": best_iou}, out_dir / "best.pt")

            model.train()

    print("\nScale training complete! Launching 100-sample test benchmark evaluation...")

    # Step 3: Run benchmark evaluation CLI
    best_pt = out_dir / "best.pt"
    eval_cmd = [
        str(ROOT / "controlled_localization/.venv/Scripts/python.exe"),
        "-m", "controlled_localization.eval_benchmark_comparison",
        "--checkpoint", str(best_pt),
        "--out-dir", str(ROOT / "controlled_localization/reports/scale_multigen_eval"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "controlled_localization/src")
    subprocess.run(eval_cmd, env=env, check=True)

    # Step 4: Write summary email report file
    summary_md = ROOT / "controlled_localization/reports/scale_multigen_eval/opensdid_benchmark_comparison.md"
    content = summary_md.read_text(encoding="utf-8") if summary_md.exists() else "Benchmark evaluation completed."

    email_body = f"""Subject: [FDDM Benchmark Report] Controlled Localization Scale Training Results

Dear Researcher,

The autonomous scale training and 100-sample OpenSDID test benchmark evaluation has successfully concluded.

{content}

Best regards,
Antigravity Research Agent
"""
    email_path = ROOT / "controlled_localization/reports/FINAL_EMAIL_REPORT.txt"
    email_path.write_text(email_body, encoding="utf-8")
    print(f"Final email report saved to: {email_path}")


if __name__ == "__main__":
    main()
