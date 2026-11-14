"""Build matched four-view controls CLI for Controlled Localization v1.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import yaml
from PIL import Image

from .data.schema import load_manifest, save_manifest, ManifestRecord
from .data.canonicalize import apply_recipe, load_canonical_rgb, SUPPORTED_RECIPES
from .utils import compute_sha256


def build_controls_for_manifest(
    records: List[ManifestRecord],
    config: Dict[str, Any],
    out_dir: Path,
    data_root: Optional[Path] = None,
) -> List[ManifestRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    recipes = config.get("recipes", ["jpeg_q70", "jpeg_q85", "jpeg_q95", "blur_s05"])
    all_output_records: List[ManifestRecord] = []

    for rec in records:
        all_output_records.append(rec)  # Keep original view (A or C)

        # Generate matched control for each recipe
        img_p = Path(rec.image_path)
        if not img_p.is_absolute() and data_root is not None:
            img_p = data_root / img_p

        img, meta = load_canonical_rgb(img_p)

        for r_id in recipes:
            degraded, r_params = apply_recipe(img, r_id)
            ctrl_role = "B" if rec.view_role == "A" else "D"
            ctrl_sample_id = f"{rec.sample_id}_{r_id}_{ctrl_role}"
            out_img_path = images_dir / f"{ctrl_sample_id}.png"
            degraded.save(out_img_path, format="PNG")

            ctrl_sha256 = compute_sha256(out_img_path)

            ctrl_record = ManifestRecord(
                schema_version=rec.schema_version,
                sample_id=ctrl_sample_id,
                source_id=rec.source_id,
                source_group_id=rec.source_group_id,
                pair_group_id=rec.pair_group_id,
                split=rec.split,
                view_role=ctrl_role,
                image_path=str(out_img_path.resolve()),
                mask_path=rec.mask_path,
                ignore_mask_path=rec.ignore_mask_path,
                label=rec.label,
                manipulation_family=rec.manipulation_family,
                editor_name=rec.editor_name,
                editor_version=rec.editor_version,
                edit_type=rec.edit_type,
                compositing_mode=rec.compositing_mode,
                processing_recipe_id=r_id,
                recipe_parameters=r_params,
                width=rec.width,
                height=rec.height,
                mask_area_fraction=rec.mask_area_fraction,
                image_sha256=ctrl_sha256,
                mask_sha256=rec.mask_sha256,
                source_dataset=rec.source_dataset,
                license_or_usage_notes=rec.license_or_usage_notes,
                mask_verified=rec.mask_verified,
                status=rec.status,
                geometry_verified=rec.geometry_verified,
                outside_pair_eligible=rec.outside_pair_eligible,
                provenance_notes=f"Generated view {ctrl_role} using recipe {r_id} from {rec.sample_id}",
            )
            all_output_records.append(ctrl_record)

    save_manifest(all_output_records, out_dir / "controls_manifest.jsonl")
    return all_output_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched controls CLI")
    parser.add_argument("--manifest", type=str, required=True, help="Path to input manifest")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--data-root", type=str, default=None, help="Root directory for relative paths")
    args = parser.parse_args()

    records = load_manifest(args.manifest)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_dir = Path(args.out)
    data_root = Path(args.data_root) if args.data_root else None

    built = build_controls_for_manifest(records, config, out_dir, data_root=data_root)
    print(f"Generated {len(built)} total records (including matched controls) in {out_dir}")


if __name__ == "__main__":
    main()

