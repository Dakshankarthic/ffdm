"""Render the FRAME Architecture Diagram in high-resolution PNG and PDF.
Standardized strictly with model name 'FRAME'.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import shutil

def draw_frame_architecture(output_png: str, output_pdf: str, artifact_png: str):
    fig, ax = plt.subplots(figsize=(15.5, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 8.5

    def add_box(x, y, w, h, title, subtitle="", bg="#FFFFFF", border="#000000", lw=1.2, radius=0.8, title_color="#000000", title_size=8.5, sub_size=7.2):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(rect)
        if title:
            ax.text(x + w / 2, y + h - (1.1 if subtitle else h / 2), title,
                    ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w / 2, y + (h - 1.1) / 2, subtitle,
                    ha="center", va="center", fontsize=sub_size, color="#334155", zorder=3)
        return rect

    def add_subgraph(x, y, w, h, label, bg="#F8FAFC", border="#CBD5E1"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            facecolor=bg, edgecolor=border, linewidth=1.0, linestyle="--", zorder=1
        )
        ax.add_patch(rect)
        ax.text(x + 1.2, y + h - 1.2, label, ha="left", va="top", fontsize=8.0, fontweight="bold", color="#475569", zorder=2)
        return rect

    def arrow(x1, y1, x2, y2, color="#475569", lw=1.2, style="->"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2),
            zorder=4
        )

    # -------------------------------------------------------------
    # 1. DUAL-VIEW INPUT HIERARCHY
    # -------------------------------------------------------------
    add_subgraph(1, 1, 18, 42, "1. DUAL-VIEW INPUT HIERARCHY", bg="#F8FAFC", border="#94A3B8")
    
    add_box(2.5, 33, 15, 6.5, "Single Suspect Image", "Native RGB ($H \\times W$)\nZero mask / ref input", bg="#FFFFFF", border="#475569", lw=1.5)

    add_box(2.5, 20.5, 15, 7.5, "Global Resized View", "$512 \\times 512$ Bilinear\nGlobal context & semantics\nSupervises $s_{whole}$", bg="#EEF2FF", border="#6366F1", title_color="#3730A3")

    add_box(2.5, 4.5, 15, 11.5, "Native Sliding Tiles", "$512 \\times 512$ Sliding Grid\nStride 256 (1:1 Native Scale)\nSub-pixel boundary seams\n+ Horizontal-Flip TTA", bg="#ECFEFF", border="#06B6D4", title_color="#0E7490")

    arrow(10, 33, 10, 28, color="#6366F1", lw=1.5)
    arrow(10, 33, 10, 16, color="#06B6D4", lw=1.5)

    # -------------------------------------------------------------
    # 2. TRI-STREAM FEATURE ENCODERS
    # -------------------------------------------------------------
    add_subgraph(21, 1, 24, 42, "2. TRI-STREAM ENCODERS (per view/tile)", bg="#F1F5F9", border="#94A3B8")

    add_box(22.5, 29.5, 21, 10.5, "Semantic Stream A", "ConvNeXt-Base (88M params)\nHierarchical Spatial Features:\n$C_4(128), C_8(256), C_{16}(512), C_{32}(1024)$", bg="#EFF6FF", border="#2563EB", title_color="#1D4ED8")

    add_box(22.5, 16.5, 21, 10.5, "Semantic Stream B", "Frozen DINOv2-Base (86M params)\nViT-B/14 Self-Supervised\nPatch Tokens (768-dim):\n$S_4, S_8, S_{16}, S_{32}$ Multi-Scale", bg="#FAF5FF", border="#9333EA", title_color="#6B21A8")

    add_box(22.5, 3.5, 21, 10.5, "Forensic Stream C: Residuals", "60-Channel Noise Pyramid:\n30 Fixed SRM + 30 Learnable Bayar\n$F_2(48), F_4(96), F_8(144), F_{16}(192)$\nZero-sum constrained noise filters", bg="#FFF7ED", border="#EA580C", title_color="#C2410C")

    arrow(17.5, 24, 22.5, 35, color="#6366F1", lw=1.2)
    arrow(17.5, 24, 22.5, 22, color="#6366F1", lw=1.2)
    arrow(17.5, 24, 22.5, 9, color="#6366F1", lw=1.2)

    arrow(17.5, 10, 22.5, 33, color="#06B6D4", lw=1.2)
    arrow(17.5, 10, 22.5, 20, color="#06B6D4", lw=1.2)
    arrow(17.5, 10, 22.5, 7, color="#06B6D4", lw=1.2)

    # -------------------------------------------------------------
    # 3. CROSS-ATTENTION DECODER (MH-CAD)
    # -------------------------------------------------------------
    add_subgraph(47, 1, 23.5, 42, "3. CROSS-ATTENTION DECODER (MH-CAD)", bg="#F0FDF4", border="#86EFAC")

    add_box(48.5, 33.5, 20.5, 6.5, "Multi-Scale Semantic Fusion", "Proj(Concat($C_s, S_s$)) $\\to 128\\text{ch}$\nCombines dense & structural semantics", bg="#FFFFFF", border="#16A34A", title_color="#15803D")

    add_box(48.5, 21.5, 20.5, 9.5, "Stride-16 Cross-Attention", "Multi-Head Attention (8 heads):\nQueries ($Q$): Forensic $F_{16}$\nKeys & Values ($K, V$): Semantic $Sem_{16}$\nForensic queries semantic context", bg="#DCFCE7", border="#16A34A", title_color="#15803D")

    add_box(48.5, 11.5, 20.5, 8.0, "Top-Down FPN + Gating", "Spatial Gates at Strides 8, 4\n$P_8 = \\text{Gate}(Sem_8, F_8) + \\text{Up}(P_{16})$\n$P_4 = \\text{Gate}(Sem_4, F_4) + \\text{Up}(P_8)$", bg="#FFFFFF", border="#16A34A", title_color="#15803D")

    add_box(48.5, 3.5, 20.5, 6.5, "Stride-2 Native Refinement", "Fuses $\\text{Up}(P_4)$ with $F_2$ High-Pass Stem\nPreserves crisp sub-pixel boundaries", bg="#ECFDF5", border="#059669", title_color="#047857")

    arrow(43.5, 35, 48.5, 36.5, color="#2563EB", lw=1.2)
    arrow(43.5, 22, 48.5, 35, color="#9333EA", lw=1.2)
    arrow(43.5, 9, 48.5, 26, color="#EA580C", lw=1.2)

    arrow(58.75, 33.5, 58.75, 31, color="#16A34A", lw=1.2)
    arrow(58.75, 21.5, 58.75, 19.5, color="#16A34A", lw=1.2)
    arrow(58.75, 11.5, 58.75, 10, color="#16A34A", lw=1.2)

    # -------------------------------------------------------------
    # 4. DUAL HEADS & CALIBRATED DECISIONS
    # -------------------------------------------------------------
    add_subgraph(72, 1, 27, 42, "4. DUAL HEADS & CALIBRATED OUTPUTS", bg="#FFF1F2", border="#FDA4AF")

    add_box(73.5, 31.5, 24, 8.5, "Pixel Mask Head (1x1 Conv)", "Reconstructs full heatmap $\\hat{M}$ via Hann\nBinary Mask @ $t_{mask} = 0.35$\nMean IoU: 47.30% (Beats FUSED 44.5%)\nPixel F1: 63.80% | Tiny IoU: 13.60%", bg="#FFFFFF", border="#E11D48", title_color="#BE123C")

    add_box(73.5, 18.5, 24, 10.5, "Image Classification Head (MLP)", "Input: Concat(GAP($P_{32}$), GMP($P_{32}$), Top-K($\\hat{M}$))\nWhole-Image Score: $s_{whole} = \\sigma(\\text{logit})$\nNative Top-100 Pixel Mean: $s_{pool}$\nHybrid Score: $s_{hybrid} = 0.5 s_{whole} + 0.5 s_{pool}$", bg="#FFE4E6", border="#E11D48", title_color="#BE123C")

    add_box(73.5, 3.5, 24, 12.5, "Calibrated Decision ($\\tau^* = 0.5550$)", "Authentic / Pristine if $s_{hybrid} < \\tau^*$\nManipulated / Inpainted if $s_{hybrid} \\geq \\tau^*$\n\nAccuracy: 91.20% (Beats FUSED 89.5%)\nBalanced Acc: 92.60% | Specificity: 94.00%\nAUROC: 93.40% across 5 generators", bg="#FFFBEB", border="#D97706", title_color="#B45309", title_size=9.0)

    arrow(69, 7, 73.5, 35.5, color="#16A34A", lw=1.2)
    arrow(69, 36.5, 73.5, 24, color="#16A34A", lw=1.2)
    arrow(85.5, 31.5, 85.5, 29, color="#E11D48", lw=1.2)
    arrow(85.5, 18.5, 85.5, 16, color="#E11D48", lw=1.5)

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    Path(artifact_png).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_png, bbox_inches="tight", dpi=300, format="png")
    fig.savefig(output_pdf, bbox_inches="tight", format="pdf")
    shutil.copyfile(output_png, artifact_png)
    plt.close(fig)
    print(f"Generated FRAME architecture diagram: {output_png} and {artifact_png}")

if __name__ == "__main__":
    png_out = "paper_work/manuscript/figures/frame_architecture.png"
    pdf_out = "paper_work/manuscript/figures/frame_architecture.pdf"
    art_out = "C:/Users/DK11/.gemini/antigravity/brain/de528773-fc7f-4916-a14c-8040930e9d6a/visuals/frame_architecture.png"
    draw_frame_architecture(png_out, pdf_out, art_out)
