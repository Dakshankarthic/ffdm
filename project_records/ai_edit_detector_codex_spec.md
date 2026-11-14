# Spatially-Anchored Forensic–Semantic MoE for AI-Edited Image Detection and Tiny-Region Localization: Research Summary & Codex Implementation Spec

## 1. Executive Research Summary & Paper Positioning

The high-level pipeline of **Single Image $\rightarrow$ Detector $\rightarrow$ "Possibly AI-Edited" Score + Highlighted Suspicious Region** is formally known in computer vision as **Image Forgery Detection and Localization (IFDL)** or **AI Inpainting Detection and Localization**. While the input-to-output pipeline is established in baselines such as [TruFor (CVPR 2023)](https://arxiv.org/html/2212.10957v3), [HiFi-Net (CVPR 2023)](https://github.com/chelsea234/hifi_ifdl), [DiffForensics (CVPR 2024)](https://cvpr.thecvf.com/virtual/2024/poster/29845), [RADAR (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/36721d1209a059dcb7a090dd543f34c4-Paper-Conference.pdf), [FUSED (Aug 2026)](https://arxiv.org/html/2608.28302v1), and [ForenScope / GIFTBench (Oct 2026)](https://arxiv.org/html/2610.01778v1), **none of the existing architectures simultaneously solve both the Global VAE Shortcut Problem and the Tiny-Edit Spatial Attribution Bottleneck (<2% image area)**.

### Core Limitations in State-of-the-Art Baselines (2025–2026)

| Baseline Model | Primary Architecture | Key Strength | Critical Failure Mode (Your Paper's Research Gap) |
| :--- | :--- | :--- | :--- |
| **[FUSED (Aug 2026)](https://arxiv.org/html/2608.28302v1)** ([Code](https://github.com/AntonNuzhdin/FUSED)) | Trainable `SparseViT` + Frozen `CLIP-ConvNeXt-XXL` ($16 \times 16$) + Top-2 of 8 SwiGLU MoE | Token-wise cross-stream routing resists global VAE artifacts across unseen diffusion generators | Compresses `SparseViT` into $m=8$ global address-free query tokens and relies on a coarse $16 \times 16$ CLIP grid; localization drops below **0.25 IoU** on tiny edits (<2% area) and suffers from wrong-object attribution (0.00 IoU) |
| **[ForenScope / GIFTBench (Oct 2026)](https://arxiv.org/html/2610.01778v1)** ([PDF](https://arxiv.org/pdf/2610.01778)) | LoRA-adapted `DINOv2` + Multi-Depth LayerMixer ($74 \times 74, 37 \times 37, 19 \times 19$) + Bounded Residual FiLM | Preserves an unconditioned $74 \times 74$ high-resolution pathway for sharp boundaries and uses Soft Dice Loss + test-time Prototype Refinement | Lacks a dedicated non-semantic forensic noise stream and lacks token-level MoE routing to separate heterogeneous generator traces |
| **[Detective SAM (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/262dd62fd1bbb30d6a6b4d578f5e65ff-Paper-Conference.pdf)** ([OpenReview](https://openreview.net/forum?id=GKJHPHNFIx)) | Frozen `SAM 2` (`Segment Anything 2`) + Perturbation-Driven Forensic Stream + Adapters | Snaps localization masks cleanly to object boundaries using SAM 2's mask decoder | Predicts only binary masks without a unified MoE image-level authenticity classifier or explicit VAE-background separation |
| **[DinoLizer (Nov 2025)](https://arxiv.org/abs/2511.20722)** | `DINOv2` with Q/V LoRA + 3-Class Linear Head + Sliding Window Inference | Explicitly separates *Authentic*, *VAE-Reconstructed Background*, and *Locally Inpainted* pixels | Uses a single-scale $14 \times 14$ linear head without multi-scale boundary decoding or low-level forensic fusion |

## 2. Proposed Novel Architecture: Spatially-Anchored Multi-Scale Forensic–Semantic MoE

Your paper bridges **[FUSED (Aug 2026)](https://arxiv.org/html/2608.28302v1)** and **[ForenScope (Oct 2026)](https://arxiv.org/html/2610.01778v1)** by replacing four bottleneck components with modern, spatially grounded alternatives:

| Pipeline Stage | Legacy Component in [FUSED](https://arxiv.org/html/2608.28302v1) | Proposed Upgrade in Your Paper | Technical & Mathematical Justification |
| :--- | :--- | :--- | :--- |
| **1. Semantic Backbone** | Frozen `OpenCLIP ConvNeXt-XXL` (single $16 \times 16$ spatial grid) | **LoRA-Adapted [DINOv2 with Registers](https://github.com/facebookresearch/dinov2)** ([arXiv](https://arxiv.org/html/2309.16588v2)) or **[SigLIP 2](https://arxiv.org/abs/2502.14786)** with **Multi-Depth LayerMixer** ($74 \times 74, 37 \times 37, 19 \times 19$) | Extracts features from 4 transformer depths ($\mathcal{L}=\{8, 11, 14, 17\}$) across 3 spatial scales, keeping a high-resolution $74 \times 74$ grid so small objects (<2% area) are never blurred out |
| **2. Forensic Token Compression** | Global Cross-Attention Pooling ($m=8$ unanchored query latents $T_{\text{svit}}$) | **[Multi-Scale Deformable Attention](https://github.com/Norman-Ou/Deformable-DETR-Torch2.x-cuda12)** + **[2D RoPE (Rotary Position Embedding)](https://github.com/naver-ai/rope-vit)** ([arXiv](https://arxiv.org/abs/2403.13298)) + **Top-$K$ Anomaly Peak Anchoring** | Samples forensic tokens directly around the top-$K$ spatial anomaly peaks $(x_k, y_k)$ of the `SparseViT` feature map $F$ and encodes explicit 2D coordinates via 2D RoPE so the MoE router knows *where* the suspicious pixels are located |
| **3. Cross-Stream Fusion Layer** | Vanilla Top-2 of 8 SwiGLU MoE | **[DeepSeekMoE Architecture](https://github.com/deepseek-ai/DeepSeek-MoE)** ([arXiv](https://arxiv.org/html/2401.06066v1)) ($E_{\text{shared}}=1$ Shared Expert + $E_{\text{routed}}=8$ Fine-Grained SwiGLU Experts, top-$k=2$) | Dedicates 1 always-active Shared Expert to absorb common cross-generator artifacts (e.g., global VAE reconstruction noise) while freeing the 8 routed experts to specialize in localized boundary and semantic tampering cues |
| **4. Mask Decoder & Loss** | 2-Layer $3 \times 3$ Conv Segmentation Bridge + BCE/Edge Loss | **Bounded CLS-FiLM Coarse-to-Fine Pyramid Decoder** (or **[SAM 2 Mask Decoder](https://github.com/facebookresearch/sam2)**) + **3-Class VAE Separation + Soft Dice & Focal Loss** | Injects the global detection `CLS` token into coarse scales ($19 \times 19, 37 \times 37$) via bounded residual FiLM while leaving the $74 \times 74$ scale unconditioned, using **Soft Dice + Focal Loss** so tiny masks receive strong gradient updates |

## 3. Mathematical Formulation (Ready for Paper Methodology Section)

### 3.1 Dual-Stream Feature Extraction
Given an input RGB image $I \in \mathbb{R}^{3 \times H \times W}$ (default $512 \times 512$ or $1036 \times 1036$ with optional high-resolution crop refinement):

1. **Non-Semantic Forensic Stream (`SparseViT`):**
   A trainable 4-stage hierarchical `SparseViT` encoder extracts multi-scale manipulation traces $\{C_j\}_{j=1}^4$, fused via learnable channel weights $\gamma_j$ and Layer Normalization:
   $$F = \mathrm{LN}\left(\sum_{j=1}^4 \gamma_j \, \mathrm{Up}(C_j)\right) \in \mathbb{R}^{d_f \times H_f \times W_f}$$

2. **Spatially-Anchored Forensic Token Sampling (Replacing Unanchored Global Pooling):**
   Instead of collapsing $F$ into $m=8$ address-free global queries, a lightweight $1 \times 1$ anomaly scoring head computes a prior anomaly heatmap $A = \sigma(W_a F) \in [0,1]^{H_f \times W_f}$. We select the top-$m$ ($m=32$) spatial peak coordinates $\{(x_q, y_q)\}_{q=1}^m$ via non-maximum suppression (NMS), apply **Multi-Scale Deformable Cross-Attention** centered at those reference points, and inject **2D Rotary Position Embeddings ($\mathrm{RoPE}_{2\mathrm{D}}$)**:
   $$T_{\text{forensic}} = \mathrm{RoPE}_{2\mathrm{D}}\Big(\mathrm{DeformAttn}(Q_{\text{init}}, F, \{(x_q, y_q)\}_{q=1}^m), \{(x_q, y_q)\}_{q=1}^m\Big) \in \mathbb{R}^{m \times d}$$

3. **Multi-Depth Semantic Stream (`DINOv2-with-Registers`):**
   A frozen (or Q/V LoRA-adapted) `DINOv2-ViT-L/14-reg4` extracts patch tokens $X^{(l)}$ from four intermediate depths $\mathcal{L} = \{8, 11, 14, 17\}$. Following [ForenScope](https://arxiv.org/html/2610.01778v1), lateral $1 \times 1$ projections resize features to three spatial scales $\mathcal{S} = \{74, 37, 19\}$ and combine depths per scale via learnable softmax `LayerMixers`:
   $$F_s = \sum_{l \in \mathcal{L}} \frac{\exp(a_{l,s})}{\sum_{l' \in \mathcal{L}} \exp(a_{l',s})} U_{l,s}, \quad s \in \{74, 37, 19\}$$
   The semantic tokens from the $37 \times 37$ (or $19 \times 19$) grid are stamped with $\mathrm{RoPE}_{2\mathrm{D}}$ to form $T_{\text{sem}} \in \mathbb{R}^{n_s \times d}$.

### 3.2 Shared + Fine-Grained Sparse MoE Fusion (`DeepSeekMoE` Style)
Concatenating the spatially anchored forensic and semantic tokens into $Z = [T_{\text{forensic}}; T_{\text{sem}}] \in \mathbb{R}^{(m + n_s) \times d}$, each token $z \in \mathbb{R}^d$ passes through a **Shared + Routed SwiGLU MoE Layer** followed by Multi-Head Self-Attention (`MSA`):
$$\mathrm{MoE}(z) = e_{\text{shared}}(z) + \sum_{i \in \mathcal{K}(z)} \tilde{r}_i(z) \, e_i(z), \quad r(z) = \mathrm{softmax}(W_r z)$$
where $e_{\text{shared}}(\cdot)$ is the always-active shared SwiGLU expert, $\mathcal{K}(z)$ indexes the top-$k=2$ routed experts out of $E=8$ specialist SwiGLU experts, and $\tilde{r}_i(z) = r_i(z) / \sum_{j \in \mathcal{K}(z)} r_j(z)$.
$$Z' = \mathrm{LN}(Z + \mathrm{MoE}(\mathrm{LN}(Z))), \quad Z'' = \mathrm{LN}(Z' + \mathrm{MSA}(\mathrm{LN}(Z')))$$

### 3.3 Dual Output Heads: Authenticity Score + High-Resolution Mask
1. **Image-Level Detection Head:**
   Attention-pools $Z''$ together with the `DINOv2` global `CLS` token $z_{\text{cls}}$ to predict both a binary manipulation probability $\hat{y} \in [0,1]$ and a 3-way image/pixel class distribution (*Authentic*, *Global VAE Reconstructed Only*, *Locally Manipulated*).
2. **Coarse-to-Fine Segmentation Bridge with Unconditioned High-Res Lane:**
   The global detection context $c = \mathrm{MLP}([z_{\text{cls}}; \mathrm{Pool}(Z'')])$ modulates the coarse semantic scales ($s \in \{19, 37\}$) via bounded residual FiLM:
   $$\widetilde{F}_s = F_s + \rho_s (g_s \odot F_s + r_s), \quad \rho_{19}=1.0, \; \rho_{37}=0.25$$
   while leaving the high-resolution semantic map $F_{74}$ and high-resolution forensic map $F$ **unconditioned** so tiny boundaries are never suppressed by global classification errors. Progressive upsampling fuses $\widetilde{F}_{19} \rightarrow \widetilde{F}_{37} \rightarrow F_{74} \rightarrow F$ to output the full-resolution manipulation mask $\hat{S} \in [0,1]^{H \times W}$.

### 3.4 Multi-Task Training Objective for Tiny-Edit Sensitivity
$$\mathcal{L}_{\text{total}} = \lambda_{\text{seg}}(\mathcal{L}_{\text{Focal}} + \mathcal{L}_{\text{SoftDice}}) + \lambda_{\text{edge}}\mathcal{L}_{\text{EDG}} + \lambda_{\text{cls}}\mathcal{L}_{\text{CE}} + \lambda_{\text{MoE}}\mathcal{L}_{\text{MoE}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{peak}}\mathcal{L}_{\text{peak}}$$
* **$\mathcal{L}_{\text{SoftDice}} + \mathcal{L}_{\text{Focal}}$:** Replaces plain BCE so tiny edits (<2% image area) produce strong, scale-invariant gradients instead of being drowned out by 98%+ background pixels.
* **$\mathcal{L}_{\text{peak}}$:** Supervises the auxiliary anomaly peak scorer $A$ against downsampled ground-truth masks so the Deformable Attention reference points lock onto true edits.

## 4. Direct Copy-Paste Prompt for Codex / AI Coding Agent

Copy and paste the prompt block below directly into **Codex**, **Claude Code**, or **Cursor** to scaffold the complete PyTorch codebase:

```text
You are an expert PyTorch computer vision and multimedia forensics researcher. Build a complete, modular, runnable PyTorch repository implementing "HiRes-FUSED" (Spatially-Anchored Multi-Scale Forensic-Semantic Mixture-of-Experts for AI-Edited Image Detection and Tiny-Region Localization).

### 1. Project Goal & Input/Output Contract
Build an end-to-end neural network in PyTorch 2.x that takes a single RGB image tensor `x` of shape `(B, 3, 512, 512)` and outputs a dictionary:
- `"pred_score"`: `(B, 1)` image-level manipulation probability ("Possibly AI-Edited" score)
- `"pred_3class_logits"`: `(B, 3, H, W)` pixel-wise logits for (0: Authentic, 1: Global VAE-Reconstructed Background, 2: Localized AI Edit)
- `"pred_mask"`: `(B, 1, 512, 512)` full-resolution probability heatmap/mask highlighting the suspicious manipulated region
- `"anomaly_peaks"`: `(B, m, 2)` normalized `(x, y)` coordinates of the top-m=32 suspicious forensic anchor points
- `"aux_losses"`: dict containing `"moe_balance_loss"`, `"token_diversity_loss"`, and `"peak_supervision_loss"`

### 2. Required Directory Structure
Create the following clean repository structure:
- `hires_fused/`
  - `__init__.py`
  - `config.py` (Dataclass configuration for model dimensions, LoRA rank, MoE experts, loss weights)
  - `models/`
    - `forensic_sparsevit.py` (4-stage hierarchical SparseViT forensic encoder with multi-scale LN channel fusion `F`)
    - `deformable_rope_pooling.py` (Anomaly peak detector + NMS top-m=32 coordinate selector + Multi-Scale Deformable Cross-Attention + 2D Rotary Position Embeddings `RoPE2D`)
    - `semantic_dinov2_pyramid.py` (Frozen `facebook/dinov2-with-registers-large` via `timm` or `transformers` with optional Q/V LoRA, extracting depths [8, 11, 14, 17] and fusing them via learnable softmax `LayerMixer` into 3 spatial grids: 74x74, 37x37, 19x19)
    - `deepseek_moe_fusion.py` (Sparsely-gated Mixture-of-Experts with 1 Always-On Shared SwiGLU Expert + E=8 Routed SwiGLU Experts with Top-k=2 linear softmax routing, load-balancing loss, followed by Multi-Head Self-Attention with 2D RoPE)
    - `decoder_bridge.py` (Bounded residual FiLM conditioning from the CLS/global token onto the 19x19 and 37x37 grids, leaving the 74x74 semantic grid and high-res SparseViT map `F` unconditioned, followed by progressive Conv-GroupNorm-GELU upsampling and optional inference-time cosine Prototype Refinement)
    - `detector.py` (Top-level `HiResFUSED` `nn.Module` integrating all components plus a two-stage high-resolution zoom-in patch cropper for <2% tiny edits)
  - `losses/`
    - `forensic_losses.py` (Combined Focal Loss + Soft Dice Loss + Sobel Edge-Weighted Loss + Image Cross-Entropy + MoE Load-Balancing Loss + Cosine Diversity Loss + Peak Supervision Loss)
  - `data/`
    - `dataset.py` (Dataset loader supporting RGB images, binary masks, optional INP-X VAE-exchanged masks, and stratification by manipulated-area fraction: `<2%`, `2-10%`, `>10%`)
  - `utils/`
    - `metrics.py` (Per-image Pixel F1, IoU, Image-level ROC-AUC, F1, Accuracy at 0.5 threshold, and stratified IoU for tiny edits `<2%` area)
    - `visualize.py` (Utility that takes an input image, runs inference, and saves a visual panel showing: Input Image, "Possibly AI-Edited (XX.X%)" badge, Anomaly Peak Anchors, Predicted Heatmap, and Highlighted Suspicious Region overlay)
- `train.py` (Complete AdamW training loop with mixed precision `torch.amp`, gradient clipping, and validation logging)
- `infer_demo.py` (CLI script: `python infer_demo.py --image path/to/img.jpg --output result.png`)
- `tests/`
  - `test_forward_and_losses.py` (Unit tests verifying tensor shapes, gradient flow through MoE and deformable pooling, and loss computation on dummy batches)

### 3. Key Implementation Requirements
1. **No Missing Placeholders:** Write complete, working PyTorch code for every module. Provide a pure-PyTorch `F.grid_sample` implementation of Multi-Scale Deformable Cross-Attention so it runs on both CPU and GPU without requiring custom C++ compilation.
2. **2D RoPE (`deformable_rope_pooling.py`):** Implement exact 2D axial Rotary Position Embeddings parameterized by normalized `(x, y)` coordinates in `[0, 1]` so both the `m=32` irregularly sampled forensic peak tokens and the `37x37` semantic grid tokens share a unified 2D coordinate space before cross-stream attention.
3. **Shared + Routed SwiGLU MoE (`deepseek_moe_fusion.py`):** Implement `SwiGLUExpert(d_model, d_ffn)` and `SharedRoutedMoE(d_model, n_routed=8, top_k=2, n_shared=1)` computing exact auxiliary load-balancing loss `L_MoE = E * sum(f_i * P_i)`.
4. **Verification:** After writing all files, run `pytest tests/test_forward_and_losses.py` to verify that a forward and backward pass on a `(2, 3, 512, 512)` batch succeeds with zero shape errors and non-zero gradients on all trainable parameters.
```
