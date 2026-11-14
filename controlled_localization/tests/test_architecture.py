"""Unit tests for model architecture, frozen BatchNorm, and fixed stem."""

import pytest
import torch
import torch.nn as nn
from controlled_localization.models.rgb_encoder import FrozenBNResNet18Encoder
from controlled_localization.models.residual_encoder import FixedHighPassStem, ResidualForensicEncoder
from controlled_localization.models.decoder import FPNDecoder
from controlled_localization.models.detector import ControlledLocalizationDetector


def test_feature_shapes_384():
    """Assert every listed 384px feature shape and full-resolution output (Section 10)."""
    detector = ControlledLocalizationDetector(pretrained_rgb=False)
    detector.eval()

    x = torch.rand((2, 3, 384, 384), dtype=torch.float32)

    # Check RGB encoder shapes
    rgb_feats = detector.rgb_encoder(x)
    assert rgb_feats["R4"].shape == (2, 64, 96, 96), f"R4 shape mismatch: {rgb_feats['R4'].shape}"
    assert rgb_feats["R8"].shape == (2, 128, 48, 48), f"R8 shape mismatch: {rgb_feats['R8'].shape}"
    assert rgb_feats["R16"].shape == (2, 256, 24, 24), f"R16 shape mismatch: {rgb_feats['R16'].shape}"
    assert rgb_feats["R32"].shape == (2, 512, 12, 12), f"R32 shape mismatch: {rgb_feats['R32'].shape}"

    # Check Residual encoder shapes
    res_feats = detector.res_encoder(x)
    assert res_feats["F2"].shape == (2, 32, 192, 192), f"F2 shape mismatch: {res_feats['F2'].shape}"
    assert res_feats["F4"].shape == (2, 64, 96, 96), f"F4 shape mismatch: {res_feats['F4'].shape}"
    assert res_feats["F8"].shape == (2, 96, 48, 48), f"F8 shape mismatch: {res_feats['F8'].shape}"
    assert res_feats["F16"].shape == (2, 128, 24, 24), f"F16 shape mismatch: {res_feats['F16'].shape}"

    # Check Detector output
    out = detector(x)
    assert "mask_logits" in out
    assert out["mask_logits"].shape == (2, 1, 384, 384), f"Output shape mismatch: {out['mask_logits'].shape}"


def test_fixed_stem_zero_parameters():
    """Frozen stem has zero parameters; buffer values unchanged after updates (Section 10)."""
    stem = FixedHighPassStem()
    param_count = sum(p.numel() for p in stem.parameters())
    assert param_count == 0, f"Stem must have 0 parameters, found {param_count}"

    buffers = list(stem.buffers())
    assert len(buffers) == 1, "Stem must have 1 buffer ('kernels')"
    kernels = buffers[0]
    assert kernels.shape == (9, 1, 3, 3)

    # Initial buffer clone
    init_kernels = kernels.clone()

    # Forward pass with gradient check
    x = torch.rand((1, 3, 64, 64), requires_grad=True)
    out = stem(x)
    loss = out.sum()
    loss.backward()

    # Buffer values must be completely identical
    assert torch.equal(stem.kernels, init_kernels), "Buffer values must never change during backprop"


def test_frozen_batchnorm_stability():
    """
    RGB BN affine/stats and num_batches_tracked unchanged after 50 train steps,
    including repeated model.train() calls. Trainable convolutions get gradients. (Section 10).
    """
    encoder = FrozenBNResNet18Encoder(pretrained=False)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

    # Record initial running stats and affine weights
    initial_states = {}
    for name, m in encoder.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            initial_states[name] = {
                "running_mean": m.running_mean.clone() if m.running_mean is not None else None,
                "running_var": m.running_var.clone() if m.running_var is not None else None,
                "num_batches_tracked": m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None,
                "weight": m.weight.clone() if m.weight is not None else None,
                "bias": m.bias.clone() if m.bias is not None else None,
            }

    # Run 50 train steps with random data and repeated train() calls
    for step in range(50):
        encoder.train()  # Explicitly trigger train()
        x = torch.randn((2, 3, 64, 64))
        feats = encoder(x)
        loss = feats["R4"].sum() + feats["R32"].sum()
        loss.backward()

        # Verify conv1 has gradients
        assert encoder.conv1.weight.grad is not None
        assert torch.isfinite(encoder.conv1.weight.grad).all()

        optimizer.step()
        optimizer.zero_grad()

    # Verify that NO BatchNorm parameters or statistics moved
    for name, m in encoder.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            init = initial_states[name]
            if init["running_mean"] is not None:
                assert torch.equal(m.running_mean, init["running_mean"]), f"{name} running_mean changed!"
            if init["running_var"] is not None:
                assert torch.equal(m.running_var, init["running_var"]), f"{name} running_var changed!"
            if init["num_batches_tracked"] is not None:
                assert torch.equal(m.num_batches_tracked, init["num_batches_tracked"]), f"{name} num_batches_tracked changed!"
            if init["weight"] is not None:
                assert torch.equal(m.weight, init["weight"]), f"{name} weight changed!"
                assert not m.weight.requires_grad, f"{name} weight should not require grad!"
            if init["bias"] is not None:
                assert torch.equal(m.bias, init["bias"]), f"{name} bias changed!"
                assert not m.bias.requires_grad, f"{name} bias should not require grad!"


def test_decoder_no_accidental_bn():
    """Forensic and decoder modules contain no accidentally trainable BatchNorm (Section 10)."""
    res_enc = ResidualForensicEncoder()
    decoder = FPNDecoder()

    for name, m in res_enc.named_modules():
        assert not isinstance(m, nn.BatchNorm2d), f"Residual encoder has BatchNorm: {name}"

    for name, m in decoder.named_modules():
        assert not isinstance(m, nn.BatchNorm2d), f"Decoder has BatchNorm: {name}"


def test_checkpoint_reproducibility(tmp_path):
    """Checkpoint reload reproduces eval predictions within numerical tolerance."""
    detector = ControlledLocalizationDetector(pretrained_rgb=False)
    detector.eval()

    x = torch.rand((1, 3, 384, 384))
    with torch.no_grad():
        orig_out = detector(x)["mask_logits"]

    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": detector.state_dict()}, ckpt_path)

    reloaded = ControlledLocalizationDetector(pretrained_rgb=False)
    reloaded.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    reloaded.eval()

    with torch.no_grad():
        reloaded_out = reloaded(x)["mask_logits"]

    assert torch.allclose(orig_out, reloaded_out, atol=1e-6), "Reloaded output differs from original!"

