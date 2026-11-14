"""Tiling, coordinate schedules, and Hann-window blending for Controlled Localization v1.1."""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


def get_tile_1d_starts(dimension: int, tile_size: int = 384, stride: int = 256) -> List[int]:
    """
    Generate 1D start coordinates for sliding tiles.
    Starts are sorted unique values from regular stride positions plus max(0, dimension - tile_size).
    Never drops edges.
    """
    if dimension <= tile_size:
        return [0]

    starts = list(range(0, dimension - tile_size + 1, stride))
    last_start = dimension - tile_size
    if last_start not in starts:
        starts.append(last_start)
    return sorted(list(set(starts)))


def get_tile_grid_coords(width: int, height: int, tile_size: int = 384, stride: int = 256) -> List[Tuple[int, int]]:
    """
    Return list of (x, y) start coordinates covering the entire [height, width] image.
    """
    x_starts = get_tile_1d_starts(width, tile_size=tile_size, stride=stride)
    y_starts = get_tile_1d_starts(height, tile_size=tile_size, stride=stride)

    coords: List[Tuple[int, int]] = []
    for y in y_starts:
        for x in x_starts:
            coords.append((x, y))
    return coords


def create_hann_window_2d(tile_size: int = 384, min_weight: float = 1e-3) -> np.ndarray:
    """
    Weight window W = max(outer(hann(384), hann(384)), 1e-3), float32.
    """
    w1d = np.hanning(tile_size).astype(np.float32)
    w2d = np.outer(w1d, w1d).astype(np.float32)
    return np.maximum(w2d, min_weight)


def pad_if_needed(image: np.ndarray, tile_size: int = 384) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Pad image with edge replication if height < tile_size or width < tile_size.
    Returns (padded_image, (pad_top, pad_bottom, pad_left, pad_right)).
    """
    h, w = image.shape[:2]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)

    if pad_h == 0 and pad_w == 0:
        return image, (0, 0, 0, 0)

    # Pad with edge values (replicate padding)
    pad_top = 0
    pad_bottom = pad_h
    pad_left = 0
    pad_right = pad_w

    if image.ndim == 3:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="edge",
        )
    else:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="edge",
        )

    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def blend_tiles_probability(
    tile_logits: List[np.ndarray],
    coords: List[Tuple[int, int]],
    image_width: int,
    image_height: int,
    tile_size: int = 384,
    use_logit_blending: bool = False,
) -> np.ndarray:
    """
    Blend tile predictions into full-resolution map using Hann window weights.
    Default: sum(W * sigmoid(tile_logits)) / sum(W).
    Logit ablation: sigmoid(sum(W * tile_logits) / sum(W)).
    """
    hann_w = create_hann_window_2d(tile_size=tile_size)

    # Accumulator buffers
    padded_w = max(image_width, tile_size)
    padded_h = max(image_height, tile_size)

    numerator = np.zeros((padded_h, padded_w), dtype=np.float32)
    weight_sum = np.zeros((padded_h, padded_w), dtype=np.float32)

    for logits, (x, y) in zip(tile_logits, coords):
        # logits shape can be [tile_size, tile_size]
        if use_logit_blending:
            val = logits.astype(np.float32)
        else:
            # stable sigmoid
            val = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float32), -30.0, 30.0)))

        numerator[y : y + tile_size, x : x + tile_size] += val * hann_w
        weight_sum[y : y + tile_size, x : x + tile_size] += hann_w

    # Ensure weight_sum > 0 everywhere
    weight_sum = np.maximum(weight_sum, 1e-6)

    merged = numerator / weight_sum

    if use_logit_blending:
        merged = 1.0 / (1.0 + np.exp(-np.clip(merged, -30.0, 30.0)))

    # Crop out padding to return exact original dimensions [image_height, image_width]
    return merged[:image_height, :image_width]

