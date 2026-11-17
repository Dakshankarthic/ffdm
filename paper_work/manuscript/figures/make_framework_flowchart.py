from pathlib import Path
OUT=Path(__file__).parent
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "DejaVu Serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
W, H = 7.1, 4.8
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
FS, LW = 5.3, 0.7

def stage(x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="none", ec="black", lw=1.0))
    ax.text((x0+x1)/2, y1-0.09, title, ha="center", va="center", fontsize=6.7, fontweight="bold")

def box(cx, cy, w, h, text, rounded=False, bold=False):
    st = "round,pad=0,rounding_size=0.06" if rounded else "square,pad=0"
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle=st, fc="white", ec="black",
                                lw=1.4 if bold else LW, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, zorder=4, linespacing=1.15)
    return dict(l=(cx-w/2, cy), r=(cx+w/2, cy), t=(cx, cy+h/2), b=(cx, cy-h/2))

def arrow(*pts):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=LW, zorder=2, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle="-|>,head_length=0.28,head_width=0.12", lw=LW, color="black",
                                shrinkA=0, shrinkB=0))

def line(*pts):
    for a, b in zip(pts[:-1], pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=LW, zorder=2)

# ---- stage frames
stage(0.05, 2.40, 1.45, 4.75, "1. Dual-View Input")
stage(1.65, 2.40, 3.75, 4.75, "2. Tri-Stream Encoders")
stage(3.95, 2.40, 7.05, 4.75, "3. Fusion & Decoder")
stage(3.95, 0.05, 7.05, 2.20, "4. Heads & Ensemble (0.65 / 0.35)")
stage(0.05, 0.05, 3.75, 2.20, "5. Thresholded Outputs")

# ---- stage 1
IMG = box(0.75, 4.15, 1.05, 0.36, "Suspect image\n(H × W, native)", rounded=True)
WV  = box(0.42, 3.50, 0.62, 0.46, "Global view\n384 × 384")
NV  = box(1.08, 3.50, 0.62, 0.46, "Native tiles\n384², stride 256")
TTA = box(1.08, 2.90, 0.62, 0.30, "Flip TTA")
arrow((0.60, IMG["b"][1]), (0.42, WV["t"][1]))
arrow((0.90, IMG["b"][1]), (1.08, NV["t"][1]))
arrow(NV["b"], TTA["t"])

# ---- stage 2 (stream C on top feeds cross-attention directly)
FOR  = box(2.70, 4.15, 1.90, 0.46, "C: 30 SRM + 30 Bayar channels\nResidual pyramid F2, F4, F8, F16")
CN   = box(2.70, 3.50, 1.90, 0.46, "A: ConvNeXt (Base / Small)\nC4, C8, C16, C32")
DINO = box(2.70, 2.90, 1.90, 0.46, "B: Frozen DINOv2 (Base / Small)\nintermediate tokens")
# bus from both views into all three encoders
line(WV["b"], (WV["b"][0], 2.58), (1.55, 2.58))
line(TTA["r"], (1.55, TTA["r"][1]))
line((1.55, 2.58), (1.55, FOR["l"][1]))
for e in (FOR, CN, DINO):
    arrow((1.55, e["l"][1]), e["l"])

# ---- stage 3
PROJ = box(4.65, 3.70, 1.35, 0.46, "Semantic projection\nProj(Concat(A, B))")
MCA  = box(6.30, 3.70, 1.35, 0.46, "Base: cross-attn. at stride 16\nSmall: gated fusion")
GATE = box(6.30, 3.05, 1.35, 0.36, "Adaptive gating (strides 8, 4)")
S2   = box(4.65, 3.05, 1.35, 0.46, "Stride-2 refinement\nUp(P4) + F2 stem")
line(CN["r"], (3.85, CN["r"][1]), (3.85, 3.70))
line(DINO["r"], (3.85, DINO["r"][1]), (3.85, 3.70))
arrow((3.85, 3.70), PROJ["l"])
arrow(PROJ["r"], MCA["l"])
arrow(FOR["r"], (6.30, FOR["r"][1]), MCA["t"])
arrow(MCA["b"], GATE["t"])
arrow(GATE["l"], S2["r"])

# ---- stage 4
MH = box(4.65, 1.70, 1.35, 0.46, "Mask head / tile merge\nlogits and Hann map")
DH = box(6.30, 1.70, 1.35, 0.46, "Image head\nsemantic + mask Top-K")
SP = box(4.65, 1.05, 1.35, 0.40, "s_pool\n(top 100 map pixels)")
SW = box(6.30, 1.05, 1.35, 0.40, "s_whole")
HY = box(5.475, 0.40, 2.55, 0.36, "s_hybrid = 0.15·s_whole + 0.85·s_pool", bold=True)
arrow(S2["b"], MH["t"])
arrow(PROJ["r"], (5.55, PROJ["r"][1]), (5.55, 2.30), (6.30, 2.30), DH["t"])
arrow(MH["r"], DH["l"])
arrow(MH["b"], SP["t"]); arrow(DH["b"], SW["t"])
arrow(SP["b"], (4.65, HY["t"][1]))
arrow(SW["b"], (6.30, HY["t"][1]))

# ---- stage 5
BM  = box(1.90, 1.70, 2.50, 0.46, "Binary localization mask\nmask threshold = 0.35")
cx, cy = 3.10, 0.70
ax.add_patch(Polygon([(cx-0.45, cy), (cx, cy+0.24), (cx+0.45, cy), (cx, cy-0.24)], closed=True,
                     fc="white", ec="black", lw=LW, zorder=3))
ax.text(cx, cy, "τ* = 0.9358", ha="center", va="center", fontsize=6.2, zorder=4)
GEN = box(1.20, 1.05, 2.10, 0.46, "BELOW THRESHOLD\nClassification: pristine")
EDT = box(1.20, 0.38, 2.10, 0.46, "AT OR ABOVE THRESHOLD\nClassification: edited")
arrow(MH["l"], BM["r"])
arrow((HY["l"][0], HY["l"][1]), (4.10, 0.40), (4.10, cy), (cx+0.45, cy))
line((cx-0.45, cy), (2.40, cy))
arrow((2.40, cy), (2.40, GEN["r"][1]), GEN["r"])
arrow((2.40, cy), (2.40, EDT["r"][1]), EDT["r"])

for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 400})):
    fig.savefig(str(OUT / f'framework_flowchart.{ext}'), **kw)
