"""Benchmark evaluation with Test-Time Augmentation (TTA) and Advanced Morphological Postprocessing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from scipy import ndimage
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from controlled_localization.models.super_frame_detector import SuperFRAMEDetector
from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.policy import InferencePolicy, compute_image_score
from controlled_localization.data.canonicalize import load_canonical_rgb
from controlled_localization.data.tiles import get_tile_grid_coords, blend_tiles_probability, pad_if_needed
from controlled_localization.metrics import compute_binary_confusion, StreamingHistogramAP
from controlled_localization.utils import resolve_device


def run_inference_with_tta(
    img_path: Path,
    detector: SuperFRAMEDetector,
    policy: InferencePolicy,
    device: torch.device,
    use_tta: bool = True,
) -> Dict[str, Any]:
    start_time = time.perf_counter()
    img, meta = load_canonical_rgb(img_path)
    orig_w, orig_h = img.size
    arr_img = np.array(img, dtype=np.float32) / 255.0

    tile_size = policy.tile_size
    stride = policy.tile_stride

    def get_prob_map(arr: np.ndarray) -> np.ndarray:
        padded_img, _ = pad_if_needed(arr, tile_size=tile_size)
        padded_h, padded_w = padded_img.shape[:2]
        coords = get_tile_grid_coords(padded_w, padded_h, tile_size=tile_size, stride=stride)
        tile_logits = []
        with torch.no_grad():
            for x, y in coords:
                crop = padded_img[y : y + tile_size, x : x + tile_size]
                tensor_crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
                out = detector(tensor_crop)
                tile_logits.append(out["mask_logits"].squeeze().cpu().numpy())
        return blend_tiles_probability(
            tile_logits=tile_logits,
            coords=coords,
            image_width=orig_w,
            image_height=orig_h,
            tile_size=tile_size,
            use_logit_blending=(policy.blending_method == "logit"),
        )

    prob_orig = get_prob_map(arr_img)

    if use_tta:
        arr_flip = np.fliplr(arr_img).copy()
        prob_flip = get_prob_map(arr_flip)
        prob_flip_back = np.fliplr(prob_flip)
        merged_prob = 0.5 * (prob_orig + prob_flip_back)
    else:
        merged_prob = prob_orig

    raw_score = compute_image_score(
        prob_map=merged_prob,
        pooling_method=policy.pooling_method,
        k=policy.pool_k,
        fraction=policy.pool_fraction,
    )

    elapsed = time.perf_counter() - start_time
    return {
        "raw_score": float(raw_score),
        "merged_prob": merged_prob,
        "runtime_seconds": float(elapsed),
    }


def postprocess_mask(
    prob: np.ndarray,
    t_mask: float = 0.35,
    min_size: int = 64,
    fill_holes: bool = True,
    gate_score: float = 1.0,
    t_image_gate: float = 0.0,
) -> np.ndarray:
    """Apply morphological hole filling, small noise island removal, and dual-task gating."""
    if gate_score < t_image_gate:
        return np.zeros_like(prob, dtype=np.uint8)

    # Initial binarization
    bin_mask = (prob > t_mask).astype(np.uint8)

    if bin_mask.sum() == 0:
        return bin_mask

    # 1. Fill holes in detected regions
    if fill_holes:
        bin_mask = ndimage.binary_fill_holes(bin_mask).astype(np.uint8)

    # 2. Remove small noise components (e.g. sensor noise, compression artifacts)
    if min_size > 0:
        labeled, num_features = ndimage.label(bin_mask)
        if num_features > 0:
            sizes = ndimage.sum(bin_mask, labeled, range(1, num_features + 1))
            mask_sizes = np.array(sizes) >= min_size
            remove_labels = np.where(~mask_sizes)[0] + 1
            for lbl in remove_labels:
                bin_mask[labeled == lbl] = 0

    return bin_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--t-mask", type=float, default=0.30)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--t-image-gate", type=float, default=0.0)
    parser.add_argument("--manifest-dir", type=str, default="I:/icvgip workshop/baseline_work/data_manifests")
    parser.add_argument("--workspace-root", type=str, default="I:/icvgip workshop")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Loading SuperFRAME from: {args.checkpoint} on {device}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    is_v2 = any("cross_attn16" in k for k in sd.keys())
    if is_v2:
        print("Detected SuperFRAME v2 architecture (Cross-Attention). Instantiating SuperFRAMEV2Detector...")
        detector = SuperFRAMEV2Detector(pretrained=False).to(device)
    else:
        print("Detected SuperFRAME architecture. Instantiating SuperFRAMEDetector...")
        detector = SuperFRAMEDetector(pretrained=False).to(device)
    detector.load_state_dict(sd)
    detector.eval()

    policy = InferencePolicy(
        policy_name="tta_eval",
        blending_method="probability",
        pooling_method="top_k",
        pool_k=2048,
        t_mask=args.t_mask,
        t_image=0.50,
    )

    manifest_files = [
        "test_sd15_manifest.json",
        "test_sd2_manifest.json",
        "test_sd3_manifest.json",
        "test_sdxl_manifest.json",
        "test_flux_manifest.json",
    ]

    workspace_root = Path(args.workspace_root)
    manifest_dir = Path(args.manifest_dir)

    all_ious = []
    all_f1s = []
    all_fps = []
    gen_stats = {}

    t0 = time.time()
    for m_file in manifest_files:
        gen_name = m_file.replace("test_", "").replace("_manifest.json", "")
        with open(manifest_dir / m_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data.get("entries", [])
        g_ious, g_f1s, g_fps = [], [], []

        for entry in entries:
            label = int(entry["label"])
            img_p = workspace_root / entry["image_path"]
            if not img_p.exists():
                continue

            res = run_inference_with_tta(img_p, detector, policy, device, use_tta=args.use_tta)
            pred_prob = res["merged_prob"]

            # Apply advanced postprocessing
            pred_bin = postprocess_mask(
                pred_prob,
                t_mask=args.t_mask,
                min_size=args.min_size,
                fill_holes=args.fill_holes,
                gate_score=res["raw_score"],
                t_image_gate=args.t_image_gate,
            )

            h, w = pred_prob.shape
            mask_rel = entry.get("mask_path")
            if label == 1 and mask_rel:
                mask_p = workspace_root / mask_rel
                with Image.open(mask_p) as m_img:
                    m_gray = m_img.convert("L").resize((w, h), resample=Image.NEAREST)
                    gt_mask = (np.array(m_gray, dtype=np.float32) > 127.0).astype(np.uint8)
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

            cm = compute_binary_confusion(pred_bin, gt_mask)
            if label == 1:
                g_ious.append(cm.iou)
                g_f1s.append(cm.f1)
                all_ious.append(cm.iou)
                all_f1s.append(cm.f1)
            else:
                fp_pct = float(np.mean(pred_bin > 0)) * 100.0
                g_fps.append(fp_pct)
                all_fps.append(fp_pct)

        gen_stats[gen_name] = {
            "mean_iou": float(np.mean(g_ious)) if g_ious else 0.0,
            "mean_f1": float(np.mean(g_f1s)) if g_f1s else 0.0,
            "mean_fp": float(np.mean(g_fps)) if g_fps else 0.0,
        }

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Postprocessing Results: TTA={args.use_tta}, fill_holes={args.fill_holes}, min_size={args.min_size}, t_mask={args.t_mask}")
    print("=" * 60)
    print(f"Overall Mean IoU:   {np.mean(all_ious)*100:.2f}%")
    print(f"Overall Mean F1:    {np.mean(all_f1s)*100:.2f}%")
    print(f"Clean False Alarm:  {np.mean(all_fps):.2f}%")
    print(f"Evaluation runtime: {elapsed:.1f}s")
    for g, s in gen_stats.items():
        print(f"  {g:8s} | IoU: {s['mean_iou']*100:.2f}% | F1: {s['mean_f1']*100:.2f}% | Clean FP: {s['mean_fp']:.2f}%")


if __name__ == "__main__":
    main()
