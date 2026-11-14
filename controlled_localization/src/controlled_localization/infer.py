"""Single-image inference CLI and pipeline for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
from PIL import Image

from .data.canonicalize import load_canonical_rgb
from .data.tiles import get_tile_grid_coords, blend_tiles_probability, pad_if_needed
from .models.detector import ControlledLocalizationDetector
from .policy import InferencePolicy, compute_image_score
from .utils import resolve_device, create_heatmap_overlay


def run_single_image_inference(
    image_path: str | Path,
    detector: ControlledLocalizationDetector,
    policy: InferencePolicy,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run full-image inference via tiled Hann-window blending.
    Deployment accepts ONLY the inspected image and policy. No masks, recipes, or originals.
    """
    start_time = time.perf_counter()
    img, meta = load_canonical_rgb(image_path)
    orig_w, orig_h = img.size

    # Convert to float32 [H, W, 3] in [0, 1]
    arr_img = np.array(img, dtype=np.float32) / 255.0

    tile_size = policy.tile_size
    stride = policy.tile_stride

    # Pad if smaller than tile_size
    padded_img, (pt, pb, pl, pr) = pad_if_needed(arr_img, tile_size=tile_size)
    padded_h, padded_w = padded_img.shape[:2]

    coords = get_tile_grid_coords(padded_w, padded_h, tile_size=tile_size, stride=stride)

    detector.eval()
    dev = next(detector.parameters()).device
    tile_logits = []

    with torch.no_grad():
        for x, y in coords:
            crop = padded_img[y : y + tile_size, x : x + tile_size]
            # [1, 3, tile_size, tile_size]
            tensor_crop = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(dev)
            out = detector(tensor_crop)
            logits_tile = out["mask_logits"].squeeze().cpu().numpy()
            tile_logits.append(logits_tile)

    use_logit_blending = (policy.blending_method == "logit")
    merged_prob = blend_tiles_probability(
        tile_logits=tile_logits,
        coords=coords,
        image_width=orig_w,
        image_height=orig_h,
        tile_size=tile_size,
        use_logit_blending=use_logit_blending,
    )

    # Compute scalar image score
    raw_score = compute_image_score(
        prob_map=merged_prob,
        pooling_method=policy.pooling_method,
        k=policy.pool_k,
        fraction=policy.pool_fraction,
    )

    # Decision rule: score > t_image
    is_fake = bool(raw_score > policy.t_image) if policy.score_direction == "higher_is_fake" else bool(raw_score < policy.t_image)

    # Binary mask: prob > t_mask
    binary_mask = (merged_prob > policy.t_mask).astype(np.uint8) * 255

    elapsed = time.perf_counter() - start_time

    return {
        "raw_score": float(raw_score),
        "is_fake": is_fake,
        "merged_prob": merged_prob,
        "binary_mask": binary_mask,
        "canonical_image": img,
        "width": orig_w,
        "height": orig_h,
        "runtime_seconds": float(elapsed),
        "tile_count": len(coords),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled Localization v1.1 Inference CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True, help="Path to single input image")
    parser.add_argument("--policy", type=str, required=True, help="Path to frozen inference_policy.json")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = InferencePolicy.load(args.policy)
    device = resolve_device()

    # Load model
    detector = ControlledLocalizationDetector(pretrained_rgb=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    detector.load_state_dict(state_dict)
    detector.to(device)
    detector.eval()

    # Run inference
    res = run_single_image_inference(args.image, detector, policy, device)

    # Save outputs
    # 1. probability.npy
    np.save(out_dir / "probability.npy", res["merged_prob"])

    # 2. mask.png
    mask_img = Image.fromarray(res["binary_mask"])
    mask_img.save(out_dir / "mask.png")

    # 3. overlay.png
    overlay_img = create_heatmap_overlay(res["canonical_image"], res["merged_prob"], threshold=policy.t_mask)
    overlay_img.save(out_dir / "overlay.png")

    # 4. result.json
    result_data = {
        "image_path": str(Path(args.image).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "policy_name": policy.policy_name,
        "raw_score": res["raw_score"],
        "is_fake_prediction": res["is_fake"],
        "t_image": policy.t_image,
        "t_mask": policy.t_mask,
        "score_direction": policy.score_direction,
        "width": res["width"],
        "height": res["height"],
        "tile_count": res["tile_count"],
        "runtime_seconds": res["runtime_seconds"],
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"Inference completed in {res['runtime_seconds']:.3f}s. Score: {res['raw_score']:.4f}, Prediction: {'FAKE' if res['is_fake'] else 'REAL'}")


if __name__ == "__main__":
    main()
