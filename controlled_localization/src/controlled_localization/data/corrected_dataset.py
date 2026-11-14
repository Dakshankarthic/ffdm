"""Phase 1: Corrected OpenSDI Dataset Loader.
Fixes the crop label noise issue identified in GEMINI_10000_IMAGE_TRAINING_HANDOFF.txt:
1. Crop-level classification label (crop_label) reflects whether the crop actually contains edited pixels.
2. Whole-image label (img_label) is kept separate.
3. Strict mask validation: missing, empty, or misaligned masks fail with explicit ValueError.
4. Smart crop sampling: balances verified edit crops, genuine background crops from edited images,
   and pristine genuine crops.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CorrectedOpenSDIDataset(Dataset):
    """
    Corrected dataset for supervised inpainting detection and localization.
    Decouples local crop edit presence from global image status.
    """

    def __init__(
        self,
        records: List[Dict],
        crop_size: int = 384,
        is_train: bool = True,
        edit_crop_prob: float = 0.65,
        strict_mask_validation: bool = True,
    ) -> None:
        self.records = records
        self.crop_size = crop_size
        self.is_train = is_train
        self.edit_crop_prob = edit_crop_prob
        self.strict_mask_validation = strict_mask_validation

    def __len__(self) -> int:
        return len(self.records)

    def _validate_and_load_mask(
        self, mask_path: Optional[str | Path], img_w: int, img_h: int, sample_id: str
    ) -> np.ndarray:
        if mask_path is None or not Path(mask_path).exists():
            if self.strict_mask_validation:
                raise ValueError(f"[Provenance Error] Missing or non-existent mask for positive sample {sample_id}: {mask_path}")
            return np.zeros((img_h, img_w), dtype=np.float32)

        with Image.open(mask_path) as m_img:
            if m_img.size != (img_w, img_h):
                if self.strict_mask_validation:
                    raise ValueError(f"[Dimension Mismatch] Image {sample_id} ({img_w}x{img_h}) != Mask ({m_img.size})")
                m_img = m_img.resize((img_w, img_h), Image.NEAREST)

            m_gray = m_img.convert("L")
            m_arr = np.array(m_gray, dtype=np.float32)

            # Polarity: values > 127.5 denote manipulated pixels
            bin_mask = (m_arr > 127.5).astype(np.float32)

            if bin_mask.sum() == 0 and self.strict_mask_validation:
                raise ValueError(f"[Integrity Error] Mask for positive sample {sample_id} contains 0 edited pixels.")

            return bin_mask

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, float, float, str]:
        """
        Returns:
          rgb_tensor:  [3, H, W] in [0, 1]
          mask_tensor: [1, H, W] in {0.0, 1.0}
          crop_label:  float (1.0 if crop contains edit pixels, 0.0 otherwise)
          img_label:   float (1.0 if original image is edited, 0.0 if genuine)
          sample_id:   str
        """
        rec = self.records[idx]
        img_p = Path(rec["image"])
        mask_val = rec.get("mask")
        img_label = float(rec["label"])
        sample_id = str(rec.get("id", idx))

        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")

        # Load RGB image
        with Image.open(img_p) as img_pil:
            img_rgb = img_pil.convert("RGB")
            w, h = img_rgb.size
            img_arr = np.array(img_rgb, dtype=np.float32) / 255.0

        # Load and strictly validate mask
        if img_label == 1.0:
            mask_arr = self._validate_and_load_mask(mask_val, w, h, sample_id)
        else:
            mask_arr = np.zeros((h, w), dtype=np.float32)

        th, tw = self.crop_size, self.crop_size

        if self.is_train:
            # Training: smart crop selection
            if h >= th and w >= tw:
                if img_label == 1.0 and random.random() < self.edit_crop_prob and mask_arr.sum() > 0:
                    # Verified Edit Crop: center near manipulated region
                    ys, xs = np.nonzero(mask_arr)
                    cy = int(np.mean(ys))
                    cx = int(np.mean(xs))
                    # Add random jitter
                    jy = cy + random.randint(-48, 48)
                    jx = cx + random.randint(-48, 48)
                    y = max(0, min(h - th, jy - th // 2))
                    x = max(0, min(w - tw, jx - tw // 2))
                else:
                    # Random crop (can be background from edited image or pristine genuine)
                    y = random.randint(0, h - th)
                    x = random.randint(0, w - tw)

                crop_rgb = img_arr[y : y + th, x : x + tw]
                crop_mask = mask_arr[y : y + th, x : x + tw]
            else:
                # Resize if smaller than crop size
                res_img = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
                res_mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                crop_rgb = np.array(res_img, dtype=np.float32) / 255.0
                crop_mask = (np.array(res_mask, dtype=np.float32) > 127.5).astype(np.float32)

            # Random horizontal flip
            if random.random() > 0.5:
                crop_rgb = np.fliplr(crop_rgb).copy()
                crop_mask = np.fliplr(crop_mask).copy()

        else:
            # Validation: center crop
            if h >= th and w >= tw:
                y = (h - th) // 2
                x = (w - tw) // 2
                crop_rgb = img_arr[y : y + th, x : x + tw]
                crop_mask = mask_arr[y : y + th, x : x + tw]
            else:
                res_img = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tw, th), Image.BILINEAR)
                res_mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)
                crop_rgb = np.array(res_img, dtype=np.float32) / 255.0
                crop_mask = (np.array(res_mask, dtype=np.float32) > 127.5).astype(np.float32)

        # CRITICAL FIX: crop_label is 1.0 ONLY IF edited pixels are present in this crop!
        has_edit = float(np.sum(crop_mask) > 0)
        crop_label = 1.0 if has_edit else 0.0

        rgb_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(crop_mask).unsqueeze(0).float()

        return rgb_tensor, mask_tensor, crop_label, img_label, sample_id
