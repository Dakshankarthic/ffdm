# Changelog

## [1.1.0] - 2026-11-13
### Initial Implementation
- Revised architecture resolving F1–F12 audit findings from v1 planning review.
- High-pass fixed stem with 9 buffers (Dx, Dy, Lap) grouped across RGB.
- Frozen ResNet-18 RGB encoder (affine and running statistics frozen).
- Four-stage GroupNorm residual encoder (32, 64, 96, 128 channels).
- 64-channel FPN decoder with exact spatial size interpolation.
- Directed Bernoulli KL view consistency with detached teacher.
- Outside consistency for verified native-pixel-paste pairs with dilated exclusion.
- Hann-windowed sliding tile inference at 384x384 with stride 256.
- Full manifest schema validation with JSONL records.

