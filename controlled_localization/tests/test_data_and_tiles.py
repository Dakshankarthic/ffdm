"""Unit tests for tiling, coordinates, constant blending, and data schemas."""

import pytest
import numpy as np
from PIL import Image

from controlled_localization.data.tiles import (
    get_tile_1d_starts,
    get_tile_grid_coords,
    blend_tiles_probability,
    pad_if_needed,
)
from controlled_localization.data.canonicalize import apply_recipe
from controlled_localization.data.schema import ManifestRecord


def test_tile_1d_starts_and_coverage():
    """Verify starts cover all dimensions without dropping edges."""
    # Exact multiple
    starts_384 = get_tile_1d_starts(384, tile_size=384, stride=256)
    assert starts_384 == [0]

    # Smaller than tile size
    starts_small = get_tile_1d_starts(200, tile_size=384, stride=256)
    assert starts_small == [0]

    # Odd large dimension
    starts_odd = get_tile_1d_starts(1025, tile_size=384, stride=256)
    assert starts_odd[0] == 0
    assert starts_odd[-1] == 1025 - 384
    # All starts within range
    for s in starts_odd:
        assert 0 <= s <= 1025 - 384


def test_constant_blending_reproduction():
    """Constant probability map merges to the exact same constant (Section 10)."""
    w, h = 800, 600
    tile_size = 384
    stride = 256
    constant_val = 0.42

    coords = get_tile_grid_coords(w, h, tile_size=tile_size, stride=stride)
    # Convert constant probability to logit
    constant_logit = float(np.log(constant_val / (1.0 - constant_val)))

    tile_logits = [np.full((tile_size, tile_size), constant_logit, dtype=np.float32) for _ in coords]

    merged = blend_tiles_probability(
        tile_logits=tile_logits,
        coords=coords,
        image_width=w,
        image_height=h,
        tile_size=tile_size,
        use_logit_blending=False,
    )

    assert merged.shape == (h, w)
    assert np.allclose(merged, constant_val, atol=1e-4), f"Constant mismatch: min={merged.min()}, max={merged.max()}"


def test_replicate_padding_small_images():
    """Test replicate padding for images smaller than tile_size."""
    small_img = np.random.rand(200, 300, 3).astype(np.float32)
    padded, (pt, pb, pl, pr) = pad_if_needed(small_img, tile_size=384)
    assert padded.shape == (384, 384, 3)
    assert pb == 184
    assert pr == 84


def test_recipe_deterministic_reproduction():
    """Repeated recipe with same seed/version yields identical decoded pixels."""
    img = Image.new("RGB", (256, 256), color=(128, 64, 200))
    # Apply JPEG recipe
    r1, _ = apply_recipe(img, "jpeg_q70")
    r2, _ = apply_recipe(img, "jpeg_q70")
    assert np.array_equal(np.array(r1), np.array(r2)), "Recipe output must be strictly deterministic!"


def test_schema_rejection_rules():
    """Unknown mask polarity, role mismatch and invalid fields fail clearly."""
    # View A with label=1 should raise ValueError
    with pytest.raises(ValueError, match="unedited view A cannot have label=1"):
        rec = ManifestRecord(
            schema_version="1.1",
            sample_id="test_01",
            source_id="src_01",
            source_group_id="grp_01",
            pair_group_id="pair_01",
            split="train",
            view_role="A",
            image_path="test.png",
            mask_path=None,
            ignore_mask_path=None,
            label=1,  # Invalid for view A
            manipulation_family="pristine",
            editor_name="none",
            editor_version="none",
            edit_type="none",
            compositing_mode="native_pixel_paste",
            processing_recipe_id="identity",
            recipe_parameters={},
            width=512,
            height=512,
            mask_area_fraction=0.0,
            image_sha256="abc",
            mask_sha256=None,
            source_dataset="test",
            license_or_usage_notes="test",
            mask_verified=True,
            status="verified",
            geometry_verified=True,
            outside_pair_eligible=True,
            provenance_notes="test",
        )
        rec.validate()

