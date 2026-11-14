"""Stage 1 synthetic demo: end-to-end inference and evaluation verification."""

import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from controlled_localization.models.detector import ControlledLocalizationDetector
from controlled_localization.policy import InferencePolicy
from controlled_localization.infer import run_single_image_inference
from controlled_localization.evaluate import evaluate_dataset
from controlled_localization.data.schema import ManifestRecord, save_manifest
from controlled_localization.utils import compute_sha256, resolve_device


def run_stage1_synthetic_demo(demo_dir: Path) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    # 1. Create a dummy model checkpoint
    detector = ControlledLocalizationDetector(pretrained_rgb=False).to(device)
    ckpt_path = demo_dir / "synthetic_checkpoint.pt"
    torch.save({"model_state_dict": detector.state_dict()}, ckpt_path)

    # 2. Create inference policy
    policy = InferencePolicy(
        policy_name="synthetic_debug_policy",
        blending_method="probability",
        pooling_method="top_k",
        pool_k=2048,
        t_mask=0.50,
        t_image=0.7075,
        score_direction="higher_is_fake",
    )
    policy_path = demo_dir / "inference_policy.json"
    policy.save(policy_path)

    # 3. Create synthetic test images and masks
    img_dir = demo_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # Create 4 synthetic test samples (2 real, 2 fake)
    for i in range(4):
        is_fake = (i >= 2)
        # Create image 512x512
        arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        img_p = img_dir / f"sample_{i:02d}_img.png"
        Image.fromarray(arr).save(img_p)

        mask_p = None
        mask_frac = 0.0
        mask_sha = None

        if is_fake:
            mask_arr = np.zeros((512, 512), dtype=np.uint8)
            # 50x50 edit region in center (0.95% edit area)
            mask_arr[230:280, 230:280] = 255
            mask_p = img_dir / f"sample_{i:02d}_mask.png"
            Image.fromarray(mask_arr).save(mask_p)
            mask_frac = float((50 * 50) / (512 * 512))
            mask_sha = compute_sha256(mask_p)

        rec = ManifestRecord(
            schema_version="1.1",
            sample_id=f"synth_{i:02d}",
            source_id=f"src_{i:02d}",
            source_group_id=f"grp_{i:02d}",
            pair_group_id=f"pair_{i:02d}",
            split="test",
            view_role="C" if is_fake else "A",
            image_path=str(img_p.resolve()),
            mask_path=str(mask_p.resolve()) if mask_p else None,
            ignore_mask_path=None,
            label=1 if is_fake else 0,
            manipulation_family="ai_local" if is_fake else "pristine",
            editor_name="synthetic_fixture",
            editor_version="1.0",
            edit_type="inpainting" if is_fake else "none",
            compositing_mode="native_pixel_paste",
            processing_recipe_id="identity",
            recipe_parameters={},
            width=512,
            height=512,
            mask_area_fraction=mask_frac,
            image_sha256=compute_sha256(img_p),
            mask_sha256=mask_sha,
            source_dataset="synthetic_demo_fixture",
            license_or_usage_notes="MIT",
            mask_verified=True,
            status="verified",
            geometry_verified=True,
            outside_pair_eligible=True,
            provenance_notes="Stage 1 synthetic validation fixture",
        )
        records.append(rec)

    manifest_path = demo_dir / "synthetic_manifest.jsonl"
    save_manifest(records, manifest_path)

    # 4. Test single-image inference
    single_res = run_single_image_inference(records[2].image_path, detector, policy, device)
    assert single_res["merged_prob"].shape == (512, 512)
    assert single_res["binary_mask"].shape == (512, 512)
    assert isinstance(single_res["raw_score"], float)

    # 5. Test full dataset evaluation
    eval_out = demo_dir / "eval_results"
    metrics = evaluate_dataset(records, detector, policy, device, eval_out)

    assert "detection" in metrics
    assert "localization" in metrics
    assert (eval_out / "metrics.json").exists()
    assert (eval_out / "per_image_predictions.csv").exists()

    demo_summary = {
        "status": "PASS",
        "device": str(device),
        "test_samples": len(records),
        "single_image_runtime_seconds": single_res["runtime_seconds"],
        "dataset_avg_seconds_per_image": metrics["avg_seconds_per_image"],
        "macro_iou": metrics["localization"]["macro_iou"],
        "metrics_file": str((eval_out / 'metrics.json').resolve()),
        "predictions_file": str((eval_out / 'per_image_predictions.csv').resolve()),
    }
    with open(demo_dir / "demo_summary.json", "w", encoding="utf-8") as f:
        json.dump(demo_summary, f, indent=2)

    print(f"Stage 1 Synthetic Demo PASSED on {device}! Metrics and predictions saved to {eval_out}")


if __name__ == "__main__":
    run_stage1_synthetic_demo(Path("I:/icvgip workshop/controlled_localization/reports/synthetic_demo"))
