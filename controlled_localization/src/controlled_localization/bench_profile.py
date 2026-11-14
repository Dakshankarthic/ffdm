"""GPU hardware and memory profiling CLI for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import torch
import yaml

from .models.detector import ControlledLocalizationDetector
from .losses.segmentation import compute_supervised_segmentation_loss
from .utils import resolve_device


def profile_hardware(config: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    tile_size = config.get("tile_size", 384)
    batch_size = config.get("batch_size", 4)  # 4 views per group

    detector = ControlledLocalizationDetector(pretrained_rgb=False).to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-4)

    # Warmup
    dummy_input = torch.rand((batch_size, 3, tile_size, tile_size), device=device, dtype=torch.float32)
    dummy_mask = torch.zeros((batch_size, 1, tile_size, tile_size), device=device, dtype=torch.float32)

    for _ in range(3):
        out = detector(dummy_input)
        loss, _, _ = compute_supervised_segmentation_loss(out["mask_logits"], dummy_mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Timed benchmark
    num_iters = 10
    start_time = time.perf_counter()

    for _ in range(num_iters):
        out = detector(dummy_input)
        loss, _, _ = compute_supervised_segmentation_loss(out["mask_logits"], dummy_mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    avg_step_time = total_time / num_iters
    avg_tile_time = avg_step_time / batch_size

    peak_allocated_mb = 0.0
    peak_reserved_mb = 0.0
    if torch.cuda.is_available():
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "batch_size": batch_size,
        "tile_size": tile_size,
        "num_iters": num_iters,
        "total_seconds": float(total_time),
        "avg_step_seconds": float(avg_step_time),
        "avg_tile_seconds": float(avg_tile_time),
        "peak_vram_allocated_mb": float(peak_allocated_mb),
        "peak_vram_reserved_mb": float(peak_reserved_mb),
    }

    with open(out_dir / "profile.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GPU hardware and memory CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out)
    report = profile_hardware(config, out_dir)
    print(f"Profile complete on {report['device_name']}: {report['avg_tile_seconds']*1000:.1f}ms/tile, Peak VRAM: {report['peak_vram_allocated_mb']:.1f} MB")


if __name__ == "__main__":
    main()

