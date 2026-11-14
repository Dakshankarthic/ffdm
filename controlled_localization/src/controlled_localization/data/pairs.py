"""Four-view pair group abstractions and loading for Controlled Localization v1.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from .canonicalize import apply_recipe, load_canonical_rgb
from .schema import ManifestRecord


@dataclass
class FourViewGroup:
    group_id: str
    image_a: np.ndarray  # [H, W, 3] float32 in [0, 1] - clean real
    image_b: np.ndarray  # [H, W, 3] float32 in [0, 1] - degraded real
    image_c: np.ndarray  # [H, W, 3] float32 in [0, 1] - clean edit
    image_d: np.ndarray  # [H, W, 3] float32 in [0, 1] - degraded edit
    mask_c: np.ndarray   # [H, W] float32 in {0, 1} - ground truth edit mask
    dilated_mask_c: np.ndarray  # [H, W] float32 in {0, 1} - dilated exclusion mask for Loutside
    ignore_mask: Optional[np.ndarray]  # [H, W] float32 in {0, 1}
    recipe_id: str
    outside_pair_eligible: bool
    width: int
    height: int


def compute_dilated_mask(mask: np.ndarray, radius_px: int = 16) -> np.ndarray:
    """
    Compute binary dilation on full-image mask BEFORE crop to prevent tile-edge boundary errors.
    Uses circular/disk structuring element or square approximation.
    """
    if radius_px <= 0 or np.sum(mask > 0.5) == 0:
        return (mask > 0.5).astype(np.float32)

    # Structuring element
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    struct = (x * x + y * y <= radius_px * radius_px)
    dilated = binary_dilation(mask > 0.5, structure=struct)
    return dilated.astype(np.float32)


def build_four_view_group(
    canonical_a_path: str,
    canonical_c_path: str,
    mask_c_path: str,
    recipe_id: str,
    group_id: str,
    outside_pair_eligible: bool = True,
    dilation_radius: int = 16,
    ignore_mask_path: Optional[str] = None,
) -> FourViewGroup:
    """
    Construct a complete four-view group (A, B, C, D) from canonical A, C and recipe.
    """
    img_a, meta_a = load_canonical_rgb(canonical_a_path)
    img_c, meta_c = load_canonical_rgb(canonical_c_path)

    if (img_a.width, img_a.height) != (img_c.width, img_c.height):
        raise ValueError(f"Geometry mismatch for group {group_id}: A is {img_a.size}, C is {img_c.size}")

    w, h = img_a.width, img_a.height

    # Apply identical recipe to A and C
    img_b, _ = apply_recipe(img_a, recipe_id)
    img_d, _ = apply_recipe(img_c, recipe_id)

    # Convert to float32 numpy arrays [H, W, 3] in [0, 1]
    arr_a = np.array(img_a, dtype=np.float32) / 255.0
    arr_b = np.array(img_b, dtype=np.float32) / 255.0
    arr_c = np.array(img_c, dtype=np.float32) / 255.0
    arr_d = np.array(img_d, dtype=np.float32) / 255.0

    # Load mask
    with Image.open(mask_c_path) as m_img:
        m_gray = m_img.convert("L")
        if (m_gray.width, m_gray.height) != (w, h):
            m_gray = m_gray.resize((w, h), resample=Image.NEAREST)
        arr_mask = (np.array(m_gray, dtype=np.float32) > 127.0).astype(np.float32)

    # Compute dilation on full image
    dilated_mask = compute_dilated_mask(arr_mask, radius_px=dilation_radius)

    arr_ignore = None
    if ignore_mask_path is not None:
        with Image.open(ignore_mask_path) as ign_img:
            ign_gray = ign_img.convert("L").resize((w, h), resample=Image.NEAREST)
            arr_ignore = (np.array(ign_gray, dtype=np.float32) > 127.0).astype(np.float32)

    return FourViewGroup(
        group_id=group_id,
        image_a=arr_a,
        image_b=arr_b,
        image_c=arr_c,
        image_d=arr_d,
        mask_c=arr_mask,
        dilated_mask_c=dilated_mask,
        ignore_mask=arr_ignore,
        recipe_id=recipe_id,
        outside_pair_eligible=outside_pair_eligible,
        width=w,
        height=h,
    )

