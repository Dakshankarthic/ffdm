"""Common utilities, seeding, hashing, atomic I/O, and overlays for Controlled Localization v1.1."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image


def seed_all(seed: int = 42) -> None:
    """Set deterministic seeds across python, numpy, torch, cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sha256(path: str | Path) -> str:
    """Compute SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_device() -> torch.device:
    """Select CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_heatmap_overlay(
    image: Image.Image,
    prob_map: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.5,
) -> Image.Image:
    """
    Render visual overlay: image with red highlight on suspicious regions (prob > threshold).
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    # Resize prob_map if needed
    if prob_map.shape != (h, w):
        prob_img = Image.fromarray(prob_map).resize((w, h), resample=Image.BILINEAR)
        probs = np.array(prob_img, dtype=np.float32)
    else:
        probs = prob_map

    arr_img = np.array(img_rgb, dtype=np.float32)

    # Red tint for suspicious regions
    mask = (probs > threshold).astype(np.float32)[:, :, np.newaxis]
    red_layer = np.zeros_like(arr_img)
    red_layer[:, :, 0] = 255.0  # pure red

    blended = arr_img * (1.0 - alpha * mask) + red_layer * (alpha * mask)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(blended)

