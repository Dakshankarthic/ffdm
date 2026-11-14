"""Inference policies, score pooling, and frozen decision rules for Controlled Localization v1.1."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


def compute_image_score(
    prob_map: np.ndarray,
    pooling_method: str = "top_k",
    k: int = 2048,
    fraction: float = 0.01,
) -> float:
    """
    Compute single scalar image score from merged probability map.
    Supported pooling methods:
      - 'top_k': mean of top min(K, N) pixels (e.g. K=2048 or 4096).
      - 'top_fraction': mean of top ceil(f * N) pixels (e.g. f=0.001, 0.005, 0.01, 0.02).
    """
    flat = prob_map.ravel()
    n = len(flat)
    if n == 0:
        return 0.0

    if pooling_method == "top_k":
        pool_size = max(1, min(k, n))
    elif pooling_method == "top_fraction":
        pool_size = max(1, int(np.ceil(fraction * n)))
    else:
        raise ValueError(f"Unknown pooling method '{pooling_method}'")

    # Use partition for fast top-K extraction
    if pool_size >= n:
        return float(np.mean(flat))

    top_vals = np.partition(flat, n - pool_size)[n - pool_size :]
    return float(np.mean(top_vals))


@dataclass
class InferencePolicy:
    policy_name: str
    blending_method: str = "probability"  # "probability" or "logit"
    pooling_method: str = "top_k"         # "top_k" or "top_fraction"
    pool_k: int = 2048
    pool_fraction: float = 0.01
    t_mask: float = 0.50                  # Threshold for binary localization mask
    t_image: float = 0.7075               # Threshold for whole-image decision
    score_direction: str = "higher_is_fake"
    target_fpr: Optional[float] = None
    observed_fpr: Optional[float] = None
    tile_size: int = 384
    tile_stride: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> InferencePolicy:
        return cls(**data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> InferencePolicy:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

