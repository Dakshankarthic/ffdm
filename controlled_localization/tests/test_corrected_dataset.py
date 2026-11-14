import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import torch

from controlled_localization.data.corrected_dataset import CorrectedOpenSDIDataset


def create_dummy_image(path: Path, width: int = 600, height: int = 600, color=(128, 128, 128)):
    img = Image.new("RGB", (width, height), color)
    img.save(path)


def create_dummy_mask(path: Path, width: int = 600, height: int = 600, edit_box=(50, 50, 100, 100)):
    # 0 = pristine, 255 = edited
    arr = np.zeros((height, width), dtype=np.uint8)
    if edit_box is not None:
        x1, y1, x2, y2 = edit_box
        arr[y1:y2, x1:x2] = 255
    img = Image.fromarray(arr, mode="L")
    img.save(path)


def test_missing_mask_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_p = tmp_p / "fake.jpg"
        create_dummy_image(img_p)

        record = {
            "image": str(img_p),
            "mask": str(tmp_p / "non_existent_mask.png"),
            "label": 1.0,
            "id": "missing_mask_test",
        }

        dataset = CorrectedOpenSDIDataset([record], crop_size=384, strict_mask_validation=True)
        with pytest.raises(ValueError, match="Provenance Error"):
            _ = dataset[0]


def test_empty_mask_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_p = tmp_p / "fake.jpg"
        mask_p = tmp_p / "empty_mask.png"
        create_dummy_image(img_p)
        create_dummy_mask(mask_p, edit_box=None)

        record = {
            "image": str(img_p),
            "mask": str(mask_p),
            "label": 1.0,
            "id": "empty_mask_test",
        }

        dataset = CorrectedOpenSDIDataset([record], crop_size=384, strict_mask_validation=True)
        with pytest.raises(ValueError, match="Integrity Error"):
            _ = dataset[0]


def test_dimension_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_p = tmp_p / "fake.jpg"
        mask_p = tmp_p / "mismatch_mask.png"
        create_dummy_image(img_p, width=600, height=600)
        create_dummy_mask(mask_p, width=400, height=400)

        record = {
            "image": str(img_p),
            "mask": str(mask_p),
            "label": 1.0,
            "id": "mismatch_test",
        }

        dataset = CorrectedOpenSDIDataset([record], crop_size=384, strict_mask_validation=True)
        with pytest.raises(ValueError, match="Dimension Mismatch"):
            _ = dataset[0]


def test_crop_label_decoupling():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_p = tmp_p / "fake.jpg"
        mask_p = tmp_p / "mask.png"
        # 1000x1000 image, edit at top-left corner [0:50, 0:50]
        create_dummy_image(img_p, width=1000, height=1000)
        create_dummy_mask(mask_p, width=1000, height=1000, edit_box=(0, 0, 50, 50))

        record = {
            "image": str(img_p),
            "mask": str(mask_p),
            "label": 1.0,
            "id": "decoupling_test",
        }

        # In val mode (center crop: 384x384 in center: 308 to 692),
        # the center crop will NOT contain the top-left edit (0 to 50).
        val_dataset = CorrectedOpenSDIDataset([record], crop_size=384, is_train=False)
        rgb, mask, crop_label, img_label, _ = val_dataset[0]

        assert img_label == 1.0, "img_label must remain 1.0 for edited image"
        assert crop_label == 0.0, "crop_label must be 0.0 when crop misses edit"
        assert mask.sum() == 0.0, "mask sum in center crop should be 0"

        # In train mode with high edit_crop_prob (1.0), it should crop around the edit
        train_dataset = CorrectedOpenSDIDataset([record], crop_size=384, is_train=True, edit_crop_prob=1.0)
        rgb, mask, crop_label, img_label, _ = train_dataset[0]

        assert img_label == 1.0
        assert crop_label == 1.0, "crop_label must be 1.0 when edit is captured"
        assert mask.sum() > 0.0, "mask should have positive pixels in edit crop"


def test_genuine_sample():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_p = Path(tmpdir)
        img_p = tmp_p / "real.jpg"
        create_dummy_image(img_p, width=500, height=500)

        record = {
            "image": str(img_p),
            "mask": None,
            "label": 0.0,
            "id": "genuine_test",
        }

        dataset = CorrectedOpenSDIDataset([record], crop_size=384, is_train=True)
        rgb, mask, crop_label, img_label, _ = dataset[0]

        assert img_label == 0.0
        assert crop_label == 0.0
        assert mask.sum() == 0.0
