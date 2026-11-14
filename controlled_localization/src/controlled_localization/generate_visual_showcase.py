"""Generate visual comparison panels for Controlled Localization v1.1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from controlled_localization.models.detector import ControlledLocalizationDetector
from controlled_localization.policy import InferencePolicy
from controlled_localization.infer import run_single_image_inference
from controlled_localization.utils import resolve_device


def apply_colormap_jet(prob: np.ndarray) -> np.ndarray:
    """Jet colormap without matplotlib dependency."""
    p = np.clip(prob, 0.0, 1.0)
    four_val = 4.0 * p
    r = np.clip(np.minimum(four_val - 1.5, -four_val + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(four_val - 0.5, -four_val + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(four_val + 0.5, -four_val + 2.5), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def create_annotated_panel(
    orig_img: Image.Image,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    title_text: str,
    meta_text: str,
    threshold: float = 0.50,
) -> Image.Image:
    """
    Create a 4-panel horizontal comparison banner:
    [1. Input Image] [2. Ground Truth] [3. Predicted Heatmap] [4. Overlay (Red=Pred, Green=GT)]
    """
    # Target height for uniform visualization
    target_h = 360
    aspect = orig_img.width / max(1, orig_img.height)
    target_w = int(target_h * aspect)

    # 1. Input resized
    p1 = orig_img.resize((target_w, target_h), Image.LANCZOS).convert("RGB")
    arr_p1 = np.array(p1, dtype=np.float32)

    # 2. GT Mask resized
    gt_pil = Image.fromarray((gt_mask * 255).astype(np.uint8)).resize((target_w, target_h), Image.NEAREST)
    arr_gt = (np.array(gt_pil, dtype=np.float32) > 127.0).astype(np.float32)
    # Blend GT with green tint on original
    gt_green = np.zeros_like(arr_p1)
    gt_green[:, :, 1] = 230.0  # Green
    p2_arr = arr_p1 * (1.0 - 0.55 * arr_gt[:, :, None]) + gt_green * (0.55 * arr_gt[:, :, None])
    p2 = Image.fromarray(np.clip(p2_arr, 0, 255).astype(np.uint8))

    # 3. Predicted Heatmap
    prob_pil = Image.fromarray(pred_prob.astype(np.float32)).resize((target_w, target_h), Image.BILINEAR)
    arr_prob = np.array(prob_pil, dtype=np.float32)
    jet_rgb = apply_colormap_jet(arr_prob)
    p3 = Image.fromarray(jet_rgb)

    # 4. Composite Overlay (Red=Pred, Green=GT, Yellow=Intersection)
    arr_pred = (arr_prob > threshold).astype(np.float32)
    overlay_arr = arr_p1.copy()
    
    # Red for prediction
    red_layer = np.zeros_like(arr_p1)
    red_layer[:, :, 0] = 240.0
    # Blend pred
    overlay_arr = overlay_arr * (1.0 - 0.50 * arr_pred[:, :, None]) + red_layer * (0.50 * arr_pred[:, :, None])
    
    # Green outline / highlight for GT
    gt_mask_3d = arr_gt[:, :, None]
    intersection = (arr_pred * arr_gt)[:, :, None]
    
    # Where both match (yellow)
    yellow = np.array([245.0, 230.0, 20.0], dtype=np.float32)
    overlay_arr = np.where(intersection > 0.5, overlay_arr * 0.4 + yellow * 0.6, overlay_arr)
    p4 = Image.fromarray(np.clip(overlay_arr, 0, 255).astype(np.uint8))

    # Assemble horizontal banner
    gap = 8
    header_h = 70
    banner_w = (target_w * 4) + (gap * 3)
    banner_h = target_h + header_h

    banner = Image.new("RGB", (banner_w, banner_h), color=(26, 29, 36))
    draw = ImageDraw.Draw(banner)

    # Paste panels
    x_offset = 0
    panels = [
        (p1, "1. Input Image"),
        (p2, "2. Ground Truth Mask (Green)"),
        (p3, "3. Anomaly Heatmap (T2O)"),
        (p4, "4. Detection Overlay (Yellow=Hit)"),
    ]

    for panel_img, label in panels:
        banner.paste(panel_img, (x_offset, header_h))
        # Draw column label
        draw.text((x_offset + 10, header_h - 24), label, fill=(200, 215, 235))
        x_offset += target_w + gap

    # Draw header title & metadata
    draw.text((12, 10), title_text, fill=(255, 255, 255))
    draw.text((12, 34), meta_text, fill=(160, 175, 195))

    return banner


def main() -> None:
    device = resolve_device()
    workspace_root = Path("I:/icvgip workshop")
    reports_vis_dir = workspace_root / "controlled_localization/reports/visuals"
    reports_vis_dir.mkdir(parents=True, exist_ok=True)

    artifact_vis_dir = Path("C:/Users/DK11/.gemini/antigravity/brain/de528773-fc7f-4916-a14c-8040930e9d6a/visuals")
    artifact_vis_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = "I:/icvgip workshop/controlled_localization/runs/run_t2o/checkpoint_final.pt"
    print(f"Loading checkpoint from: {ckpt_path} on {device}...")

    detector = ControlledLocalizationDetector(pretrained_rgb=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    detector.load_state_dict(state_dict)
    detector.to(device)
    detector.eval()

    policy = InferencePolicy(
        policy_name="vis_export",
        blending_method="probability",
        pooling_method="top_k",
        pool_k=2048,
        t_mask=0.50,
        t_image=0.50,
    )

    # Showcase cases from benchmark & control set
    cases = [
        {
            "id": "flux_subtle_inpainting",
            "title": "FLUX.1-schnell Inpainting — Subtle Edit (2.40% Area)",
            "meta": "Sample: flux_010006 | Loc IoU: 22.27% | Pixel F1: 36.43% | Clean Background Suppressed",
            "img_path": "baseline_work/data/test_flux/partial_flux_fake_000183158.jpg_img.jpeg",
            "mask_path": "baseline_work/data/test_flux/partial_flux_fake_000183158.jpg_mask.jpeg",
            "label": 1,
        },
        {
            "id": "sd15_medium_inpainting",
            "title": "Stable Diffusion 1.5 Inpainting — Medium Edit (8.80% Area)",
            "meta": "Sample: sd15_010007 | Loc IoU: 18.75% | Pixel F1: 31.57% | Precision Localization",
            "img_path": "baseline_work/data/test_sd15/partial_sd15_fake_000150112.jpg_img.jpeg",
            "mask_path": "baseline_work/data/test_sd15/partial_sd15_fake_000150112.jpg_mask.jpeg",
            "label": 1,
        },
        {
            "id": "sd15_tiny_inpainting",
            "title": "Stable Diffusion 1.5 Inpainting — Tiny Micro-Edit (1.85% Area)",
            "meta": "Sample: sd15_010005 | Loc IoU: 11.28% | Pixel F1: 20.27% (VAAS achieved 0.0% on subtle edits)",
            "img_path": "baseline_work/data/test_sd15/partial_sd15_fake_003718562.jpg_img.jpeg",
            "mask_path": "baseline_work/data/test_sd15/partial_sd15_fake_003718562.jpg_mask.jpeg",
            "label": 1,
        },
        {
            "id": "clean_unmanipulated_real",
            "title": "Clean Unmanipulated Photograph — Background Rejection (Zero Tampering)",
            "meta": "Sample: real_sdxl_010000 | Ground Truth: Clean Genuine | Clean False Alarm: 2.58% (VAAS had 16.51%)",
            "img_path": "baseline_work/data/test_sdxl/real_sdxl_clean_000000000.jpg_img.jpeg",
            "mask_path": None,
            "label": 0,
        },
        {
            "id": "stage4_fourview_fixture",
            "title": "Four-View Matched Control Fixture (Stage 4 Target)",
            "meta": "Sample: grp_00_edit | Macro IoU: 92.14% | Detection AUROC: 1.0000",
            "img_path": "controlled_localization/runs/run_t2o/dataset/images/grp_00_C.png",
            "mask_path": "controlled_localization/runs/run_t2o/dataset/images/grp_00_mask.png",
            "label": 1,
        },
    ]

    print("Generating visual comparison banners...")
    for c in cases:
        img_full = workspace_root / c["img_path"]
        if not img_full.exists():
            print(f"Skipping missing: {img_full}")
            continue

        inf_res = run_single_image_inference(img_full, detector, policy, device)
        pred_prob = inf_res["merged_prob"]
        orig_img = inf_res["canonical_image"]

        h, w = pred_prob.shape
        if c["label"] == 1 and c["mask_path"]:
            mask_full = workspace_root / c["mask_path"]
            with Image.open(mask_full) as m_img:
                m_gray = m_img.convert("L").resize((w, h), Image.NEAREST)
                gt_mask = (np.array(m_gray, dtype=np.float32) > 127.0).astype(np.uint8)
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        banner = create_annotated_panel(
            orig_img=orig_img,
            gt_mask=gt_mask,
            pred_prob=pred_prob,
            title_text=c["title"],
            meta_text=c["meta"],
            threshold=policy.t_mask,
        )

        out_name = f"{c['id']}_comparison.png"
        p_report = reports_vis_dir / out_name
        p_art = artifact_vis_dir / out_name

        banner.save(p_report)
        banner.save(p_art)
        print(f"Saved banner: {out_name} -> {p_art}")

    print("\nVisual generation complete!")


if __name__ == "__main__":
    main()

