"""Fast Feature-Cached Multi-Evidence Training and Multi-Generator Benchmark.
Pre-extracts frozen 257-dim features for whole images and verified native tiles,
trains the Multi-Evidence Combiner Head in seconds,
and evaluates across all 5 generators on the 100-sample test benchmark.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.models.multi_evidence_head import MultiEvidenceDetectionHead
from controlled_localization.data.canonicalize import load_canonical_rgb
from controlled_localization.data.tiles import get_tile_grid_coords, pad_if_needed
from controlled_localization.utils import seed_all, resolve_device


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
DEFAULT_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"
CHECKPOINT_V2 = RUNS_DIR / "super_frame_v2_run/best.pt"
CACHE_PATH = RUNS_DIR / "multi_evidence_head_v2/features_cache.pt"


def fast_sample_tiles(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    label: float,
    tile_size: int = 384,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    h, w, _ = img_arr.shape

    if h < tile_size or w < tile_size:
        res_img = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
        res_arr = np.array(res_img, dtype=np.float32) / 255.0
        return res_arr, label, res_arr, 0.0

    # Tile 1: Verified edit tile (if manipulated)
    if label == 1.0 and np.sum(mask_arr) > 0:
        ys, xs = np.nonzero(mask_arr)
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))
        jy = cy + random.randint(-16, 16)
        jx = cx + random.randint(-16, 16)
        y1 = max(0, min(h - tile_size, jy - tile_size // 2))
        x1 = max(0, min(w - tile_size, jx - tile_size // 2))
        t1 = img_arr[y1 : y1 + tile_size, x1 : x1 + tile_size]
        l1 = 1.0
    else:
        y1 = random.randint(0, h - tile_size)
        x1 = random.randint(0, w - tile_size)
        t1 = img_arr[y1 : y1 + tile_size, x1 : x1 + tile_size]
        l1 = 0.0

    # Tile 2: Genuine background tile
    if label == 1.0 and np.sum(mask_arr) > 0:
        # Try up to 5 random positions to find clean patch
        t2 = None
        for _ in range(5):
            y2 = random.randint(0, h - tile_size)
            x2 = random.randint(0, w - tile_size)
            if np.mean(mask_arr[y2 : y2 + tile_size, x2 : x2 + tile_size]) < 0.005:
                t2 = img_arr[y2 : y2 + tile_size, x2 : x2 + tile_size]
                break
        if t2 is None:
            y2 = (y1 + tile_size) % max(1, h - tile_size)
            x2 = (x1 + tile_size) % max(1, w - tile_size)
            t2 = img_arr[y2 : y2 + tile_size, x2 : x2 + tile_size]
        l2 = 0.0
    else:
        y2 = random.randint(0, h - tile_size)
        x2 = random.randint(0, w - tile_size)
        t2 = img_arr[y2 : y2 + tile_size, x2 : x2 + tile_size]
        l2 = 0.0

    return t1, l1, t2, l2


def extract_features_dataset(
    records: List[Dict],
    detector: SuperFRAMEV2Detector,
    device: torch.device,
    desc: str = "",
) -> Dict[str, torch.Tensor]:
    wholes, t1s, t2s = [], [], []
    y_imgs, y_t1s, y_t2s = [], [], []

    t0 = time.time()
    for i, r in enumerate(records):
        img_p = Path(r["image"])
        mask_val = r.get("mask")
        label = float(r["label"])

        img = Image.open(img_p).convert("RGB")
        w, h = img.size

        # Whole image resized
        img_whole = img.resize((384, 384), Image.BILINEAR)
        arr_whole = np.array(img_whole, dtype=np.float32) / 255.0

        if label == 1.0 and mask_val is not None and Path(mask_val).exists():
            mask_img = Image.open(mask_val).convert("L")
            mask_arr = (np.array(mask_img, dtype=np.float32) > 127.5).astype(np.float32)
        else:
            mask_arr = np.zeros((h, w), dtype=np.float32)

        arr_img = np.array(img, dtype=np.float32) / 255.0
        t1, l1, t2, l2 = fast_sample_tiles(arr_img, mask_arr, label, tile_size=384)

        ten_w = torch.from_numpy(arr_whole).permute(2, 0, 1).unsqueeze(0).to(device)
        ten_t1 = torch.from_numpy(t1).permute(2, 0, 1).unsqueeze(0).to(device)
        ten_t2 = torch.from_numpy(t2).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            feat_w = detector(ten_w, return_features=True)["det_feat"].cpu()
            feat_t1 = detector(ten_t1, return_features=True)["det_feat"].cpu()
            feat_t2 = detector(ten_t2, return_features=True)["det_feat"].cpu()

        wholes.append(feat_w)
        t1s.append(feat_t1)
        t2s.append(feat_t2)
        y_imgs.append(label)
        y_t1s.append(l1)
        y_t2s.append(l2)

        if (i + 1) % 50 == 0 or (i + 1) == len(records):
            print(f"  [{desc}] Cached {i+1}/{len(records)} in {time.time()-t0:.1f}s", flush=True)

    e_whole = torch.cat(wholes, dim=0)   # [N, 257]
    e_t1 = torch.cat(t1s, dim=0)         # [N, 257]
    e_t2 = torch.cat(t2s, dim=0)         # [N, 257]
    e_tiles = torch.stack([e_t1, e_t2], dim=1)  # [N, 2, 257]

    return {
        "e_whole": e_whole,
        "e_tiles": e_tiles,
        "y_img": torch.tensor(y_imgs, dtype=torch.float32).unsqueeze(1),
        "y_tiles": torch.tensor([[l1, l2] for l1, l2 in zip(y_t1s, y_t2s)], dtype=torch.float32).unsqueeze(2),
    }


def main() -> None:
    seed_all(42)
    device = resolve_device()
    save_dir = RUNS_DIR / "multi_evidence_head_v2"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================", flush=True)
    print("FAST MULTI-EVIDENCE COMBINER TRAINING (SuperFRAME v2)", flush=True)
    print("================================================================================", flush=True)

    # 1. Load frozen detector
    print(f"Loading frozen SuperFRAME v2 from: {CHECKPOINT_V2}", flush=True)
    ckpt = torch.load(CHECKPOINT_V2, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    detector = SuperFRAMEV2Detector(pretrained=False).to(device)
    detector.load_state_dict(sd)
    detector.eval()

    for p in detector.parameters():
        p.requires_grad = False

    # 2. Extract or load feature cache
    if CACHE_PATH.exists():
        print(f"Loading cached features from: {CACHE_PATH}", flush=True)
        cached = torch.load(CACHE_PATH, map_location="cpu")
        train_cache = cached["train"]
        val_cache = cached["val"]
    else:
        with open(DEFAULT_MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        train_pos = [r for r in manifest if r["split"] == "train" and r["label"] == 1][:120]
        train_neg = [r for r in manifest if r["split"] == "train" and r["label"] == 0][:120]
        train_sub = train_pos + train_neg

        val_pos = [r for r in manifest if r["split"] == "val" and r["label"] == 1][:50]
        val_neg = [r for r in manifest if r["split"] == "val" and r["label"] == 0][:50]
        val_sub = val_pos + val_neg

        print(f"Precomputing features: {len(train_sub)} train samples, {len(val_sub)} val samples...", flush=True)
        train_cache = extract_features_dataset(train_sub, detector, device, desc="TRAIN")
        val_cache = extract_features_dataset(val_sub, detector, device, desc="VAL")

        torch.save({"train": train_cache, "val": val_cache}, CACHE_PATH)
        print(f"Saved feature cache to: {CACHE_PATH}", flush=True)

    # Move tensors to GPU
    tw = train_cache["e_whole"].to(device)
    tt = train_cache["e_tiles"].to(device)
    ty = train_cache["y_img"].to(device)
    tyt = train_cache["y_tiles"].to(device)

    vw = val_cache["e_whole"].to(device)
    vt = val_cache["e_tiles"].to(device)
    vy = val_cache["y_img"].to(device)
    vyt = val_cache["y_tiles"].to(device)

    # 3. Train Head
    head = MultiEvidenceDetectionHead(in_dim=257, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_bce = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_head_path = save_dir / "best_head.pt"

    print("\nTraining Multi-Evidence Combiner Head on cached tensors...", flush=True)
    num_epochs = 30
    batch_size = 32
    num_train = tw.shape[0]

    for epoch in range(1, num_epochs + 1):
        head.train()
        indices = torch.randperm(num_train)
        epoch_loss = 0.0

        for start in range(0, num_train, batch_size):
            idx = indices[start : start + batch_size]
            b_w = tw[idx]
            b_t = tt[idx]
            b_y = ty[idx]
            b_yt = tyt[idx]

            optimizer.zero_grad()
            comb_logit, whole_logit, tile_logits = head(b_w, b_t)

            loss_c = criterion_bce(comb_logit, b_y)
            loss_w = criterion_bce(whole_logit, b_y)
            loss_t = criterion_bce(tile_logits, b_yt)

            loss = loss_c + 0.5 * loss_w + 0.5 * loss_t
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation evaluation
        head.eval()
        with torch.no_grad():
            v_comb, v_whole, _ = head(vw, vt)
            v_scores = torch.sigmoid(v_comb).cpu().numpy().flatten()
            v_targets = vy.cpu().numpy().flatten()

            auc = float(roc_auc_score(v_targets, v_scores))
            acc = float(accuracy_score(v_targets, (v_scores > 0.5).astype(int)))
            f1 = float(f1_score(v_targets, (v_scores > 0.5).astype(int), zero_division=0))

        if epoch % 5 == 0 or epoch == num_epochs or auc > best_val_auc:
            print(f"Epoch {epoch:2d}/{num_epochs:2d} - Loss: {epoch_loss:.4f} | Val AUROC: {auc:.4f} | Val Acc: {acc*100:.2f}% | Val F1: {f1*100:.2f}%", flush=True)

        if auc > best_val_auc:
            best_val_auc = auc
            torch.save({
                "head_state_dict": head.state_dict(),
                "val_auc": auc,
                "val_acc": acc,
                "epoch": epoch,
            }, best_head_path)

    print(f"\nTraining Complete. Best Validation AUROC: {best_val_auc:.4f}. Saved: {best_head_path}", flush=True)

    # 4. Evaluate on Official 100-sample multi-generator test set
    print("\n================================================================================", flush=True)
    print("EVALUATING TRAINED MULTI-EVIDENCE HEAD ON 100-SAMPLE TEST BENCHMARK", flush=True)
    print("================================================================================", flush=True)

    head.load_state_dict(torch.load(best_head_path)["head_state_dict"])
    head.eval()

    manifest_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]
    manifest_dir = ROOT / "baseline_work/data_manifests"

    all_y = []
    all_scores_comb = []
    all_scores_whole = []
    per_gen = {}

    for m_file in manifest_files:
        gen_name = m_file.replace("test_", "").replace("_manifest.json", "")
        with open(manifest_dir / m_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"  Testing generator: {gen_name.upper()} (20 samples)...", flush=True)
        gy, g_comb, g_whole = [], [], []

        for entry in data.get("entries", []):
            label = int(entry["label"])
            img_p = ROOT / entry["image_path"]
            if not img_p.exists():
                continue

            img, _ = load_canonical_rgb(img_p)
            orig_w, orig_h = img.size
            arr_img = np.array(img, dtype=np.float32) / 255.0

            # Whole image
            img_w = img.resize((384, 384), Image.BILINEAR)
            arr_w = np.array(img_w, dtype=np.float32) / 255.0
            ten_w = torch.from_numpy(arr_w).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                out_w = detector(ten_w, return_features=True)
                e_w = out_w["det_feat"]

            # Native tiles
            padded_img, _ = pad_if_needed(arr_img, tile_size=384)
            ph, pw = padded_img.shape[:2]
            coords = get_tile_grid_coords(pw, ph, tile_size=384, stride=256)

            tile_feats = []
            with torch.no_grad():
                for x, y in coords:
                    crop = padded_img[y : y + 384, x : x + 384]
                    ten_c = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
                    out_c = detector(ten_c, return_features=True)
                    tile_feats.append(out_c["det_feat"])

            e_tiles = torch.cat(tile_feats, dim=0).unsqueeze(0)  # [1, N, 257]

            with torch.no_grad():
                comb_l, whole_l, _ = head(e_w, e_tiles)
                sc = float(torch.sigmoid(comb_l).item())
                sw = float(torch.sigmoid(whole_l).item())

            gy.append(label)
            g_comb.append(sc)
            g_whole.append(sw)
            all_y.append(label)
            all_scores_comb.append(sc)
            all_scores_whole.append(sw)

        def calc_gen(y: List[int], s: List[float]) -> Dict[str, float]:
            y_a = np.array(y)
            s_a = np.array(s)
            auc = float(roc_auc_score(y_a, s_a)) if len(np.unique(y_a)) > 1 else 0.5
            preds = (s_a > 0.5).astype(int)
            acc = float(accuracy_score(y_a, preds))
            bal_acc = float(balanced_accuracy_score(y_a, preds)) if len(np.unique(y_a)) > 1 else 0.5
            f1 = float(f1_score(y_a, preds, zero_division=0))
            return {"roc_auc": round(auc, 4), "acc": round(acc, 4), "bal_acc": round(bal_acc, 4), "f1": round(f1, 4)}

        per_gen[gen_name] = {
            "combined": calc_gen(gy, g_comb),
            "whole": calc_gen(gy, g_whole),
        }

    y_all = np.array(all_y)
    sc_all = np.array(all_scores_comb)
    sw_all = np.array(all_scores_whole)

    def calc_tot(s: np.ndarray) -> Dict[str, float]:
        auc = float(roc_auc_score(y_all, s))
        preds = (s > 0.5).astype(int)
        acc = float(accuracy_score(y_all, preds))
        bal_acc = float(balanced_accuracy_score(y_all, preds))
        f1 = float(f1_score(y_all, preds, zero_division=0))
        return {"roc_auc": round(auc, 4), "acc": round(acc, 4), "bal_acc": round(bal_acc, 4), "f1": round(f1, 4)}

    final_results = {
        "overall": {
            "combined": calc_tot(sc_all),
            "whole": calc_tot(sw_all),
        },
        "per_generator": per_gen,
    }

    out_json = ROOT / "controlled_localization/reports/trained_multi_evidence_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print("\n" + "=" * 90, flush=True)
    print("FINAL BENCHMARK COMPARISON TABLE (SUPERFRAME v2 MULTI-EVIDENCE)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Method':<25} | {'ROC-AUC':<10} | {'Accuracy (%)':<14} | {'Bal Acc (%)':<14} | {'F1 (%)'}", flush=True)
    print("-" * 90, flush=True)
    m_c = final_results["overall"]["combined"]
    m_w = final_results["overall"]["whole"]
    print(f"{'Multi-Evidence Combined':<25} | {m_c['roc_auc']:<10.4f} | {m_c['acc']*100:<14.2f} | {m_c['bal_acc']*100:<14.2f} | {m_c['f1']*100:.2f}%", flush=True)
    print(f"{'Whole-Image Only':<25} | {m_w['roc_auc']:<10.4f} | {m_w['acc']*100:<14.2f} | {m_w['bal_acc']*100:<14.2f} | {m_w['f1']*100:.2f}%", flush=True)
    print("-" * 90, flush=True)

    print("\nPER-GENERATOR ACCURACY & F1 BREAKDOWN:", flush=True)
    print(f"{'Generator':<12} | {'ROC-AUC':<10} | {'Accuracy (%)':<14} | {'Bal Acc (%)':<14} | {'F1 (%)'}", flush=True)
    print("-" * 75, flush=True)
    for g, data in per_gen.items():
        m = data["combined"]
        print(f"{g.upper():<12} | {m['roc_auc']:<10.4f} | {m['acc']*100:<14.2f} | {m['bal_acc']*100:<14.2f} | {m['f1']*100:.2f}%", flush=True)
    print("-" * 75, flush=True)
    print(f"Complete results saved to: {out_json}", flush=True)


if __name__ == "__main__":
    main()
