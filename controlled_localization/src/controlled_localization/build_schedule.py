"""Build deterministic training schedule CLI for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List
import yaml

from .data.schema import load_manifest, ManifestRecord
from .data.tiles import get_tile_grid_coords


def build_schedule_file(
    manifest_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> None:
    records = load_manifest(manifest_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    rng = random.Random(seed)
    num_steps = config.get("train", {}).get("steps", 5000)
    recipes = config.get("recipes", ["jpeg_q70", "jpeg_q85", "jpeg_q95", "blur_s05", "identity"])

    # Group records by source_group_id
    groups: Dict[str, List[ManifestRecord]] = {}
    for r in records:
        if r.status == "verified":
            groups.setdefault(r.source_group_id, []).append(r)

    group_keys = sorted(list(groups.keys()))
    if not group_keys:
        raise ValueError(f"No verified groups found in {manifest_path}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", encoding="utf-8") as f:
        for step in range(num_steps):
            # Select 2 groups per step (one edit-intersecting, one uniform/background)
            g1 = rng.choice(group_keys)
            g2 = rng.choice(group_keys)
            recipe1 = rng.choice(recipes)
            recipe2 = rng.choice(recipes)

            entry = {
                "step": step,
                "group1": {
                    "source_group_id": g1,
                    "crop_type": "edit_intersecting",
                    "crop_x": rng.randint(0, 100),
                    "crop_y": rng.randint(0, 100),
                    "recipe_id": recipe1,
                },
                "group2": {
                    "source_group_id": g2,
                    "crop_type": "uniform",
                    "crop_x": rng.randint(0, 100),
                    "crop_y": rng.randint(0, 100),
                    "recipe_id": recipe2,
                },
                "seed": rng.randint(0, 1000000),
            }
            f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible training schedule CLI")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out", type=str, required=True, help="Output path for schedule.jsonl")
    args = parser.parse_args()

    build_schedule_file(args.manifest, args.config, args.out)
    print(f"Schedule generated at {args.out}")


if __name__ == "__main__":
    main()

