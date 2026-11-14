"""Image canonicalization and degradation recipe generation for Controlled Localization v1.1."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from PIL import Image, ImageFilter, ImageOps


SUPPORTED_RECIPES = {
    "identity": {"description": "Lossless identity passthrough"},
    "jpeg_q70": {"quality": 70, "subsampling": 2, "description": "JPEG Quality 70 4:2:0"},
    "jpeg_q85": {"quality": 85, "subsampling": 2, "description": "JPEG Quality 85 4:2:0"},
    "jpeg_q95": {"quality": 95, "subsampling": 2, "description": "JPEG Quality 95 4:2:0"},
    "blur_s05": {"radius": 0.5, "description": "Gaussian blur sigma=0.5"},
}


def load_canonical_rgb(image_path: str | Path) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Load image, apply EXIF orientation once, convert to sRGB 8-bit, and return metadata.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(path) as img:
        orig_format = img.format
        orig_size = img.size
        # Apply EXIF transpose if available
        transposed = ImageOps.exif_transpose(img)
        if transposed is None:
            transposed = img.copy()
        rgb_img = transposed.convert("RGB")

    metadata = {
        "original_format": orig_format,
        "original_width": orig_size[0],
        "original_height": orig_size[1],
        "canonical_width": rgb_img.width,
        "canonical_height": rgb_img.height,
        "mode": "RGB",
    }
    return rgb_img, metadata


def apply_recipe(image: Image.Image, recipe_id: str, seed: int = 42) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Apply a deterministic recipe to an image.
    Recipe is applied to the full image BEFORE cropping tiles to preserve block phase.
    """
    if recipe_id not in SUPPORTED_RECIPES:
        raise ValueError(f"Unknown recipe_id '{recipe_id}'. Supported: {list(SUPPORTED_RECIPES.keys())}")

    recipe_info = SUPPORTED_RECIPES[recipe_id]

    if recipe_id == "identity":
        return image.copy(), {"recipe_id": recipe_id, "params": {}}

    elif recipe_id.startswith("jpeg_"):
        quality = recipe_info["quality"]
        subsampling = recipe_info["subsampling"]  # 2 corresponds to 4:2:0 in libjpeg
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
        buffer.seek(0)
        degraded = Image.open(buffer).convert("RGB")
        return degraded, {"recipe_id": recipe_id, "quality": quality, "subsampling": "4:2:0"}

    elif recipe_id == "blur_s05":
        radius = recipe_info["radius"]
        degraded = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return degraded, {"recipe_id": recipe_id, "radius": radius}

    else:
        raise NotImplementedError(f"Recipe handler not implemented for {recipe_id}")

