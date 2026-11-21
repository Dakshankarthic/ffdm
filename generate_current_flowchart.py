import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none'
})

W, H = 7.2, 4.4
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')

FS_TITLE = 7.0
FS_BOX = 5.6
FS_SUB = 4.8
LW = 0.85

def stage(x0, y0, x1, y1, title, bg='#f8fafc'):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0, boxstyle='round,pad=0,rounding_size=0.08',
                                fc=bg, ec='#334155', lw=1.1, zorder=1))
    ax.text((x0+x1)/2, y1-0.12, title, ha='center', va='center', fontsize=FS_TITLE, fontweight='bold', color='#0f172a', zorder=2)

def box(cx, cy, w, h, title, sub='', bg='white', border='#1e293b', bold=False):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle='round,pad=0,rounding_size=0.05',
                                fc=bg, ec=border, lw=1.2 if bold else LW, zorder=3))
    if sub:
        ax.text(cx, cy+0.07, title, ha='center', va='center', fontsize=FS_BOX, fontweight='bold', color='#0f172a', zorder=4)
        ax.text(cx, cy-0.08, sub, ha='center', va='center', fontsize=FS_SUB, color='#334155', zorder=4, linespacing=1.1)
    else:
        ax.text(cx, cy, title, ha='center', va='center', fontsize=FS_BOX, fontweight='bold' if bold else 'normal', color='#0f172a', zorder=4)
    return {'l': (cx-w/2, cy), 'r': (cx+w/2, cy), 't': (cx, cy+h/2), 'b': (cx, cy-h/2)}

def arrow(*pts, color='#1e293b'):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=LW, zorder=2)
    ax.annotate('', xy=pts[-1], xytext=pts[-2], zorder=2,
                arrowprops=dict(arrowstyle='-|>,head_length=0.24,head_width=0.11', lw=LW, color=color, shrinkA=0, shrinkB=0))

def line(*pts, color='#1e293b'):
    for a, b in zip(pts[:-1], pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=LW, zorder=2)

# --- Top row stages ---
stage(0.08, 2.15, 1.70, 4.32, '1. Dual-View Input Stage')
stage(1.85, 2.15, 4.30, 4.32, '2. Shared ConvNeXt-Tiny Backbone')
stage(4.45, 2.15, 7.12, 4.32, '3. 64-ch Lateral FPN Decoder')

# --- Bottom row stages ---
stage(4.45, 0.08, 7.12, 2.02, '4. Hann Window Reconstruction')
stage(0.08, 0.08, 4.30, 2.02, '5. Dual Heads & Calibrated Verdict')

# --- Stage 1 Content ---
IMG = box(0.89, 3.82, 1.40, 0.38, 'Suspect RGB Photograph', 'H x W (Arbitrary Native Scale)')
WV  = box(0.50, 2.95, 0.72, 0.52, 'Global View', '384 x 384\nScene Context')
NV  = box(1.28, 2.95, 0.72, 0.52, 'Native Tiles', '384 x 384, s=256\n1:1 Pixel Scale')
TTA = box(1.28, 2.36, 0.72, 0.24, 'Flip TTA', '')

arrow((0.65, IMG['b'][1]), (0.50, WV['t'][1]))
arrow((1.13, IMG['b'][1]), (1.28, NV['t'][1]))
arrow(NV['b'], TTA['t'])

# --- Stage 2 Content ---
C_BACK = box(3.07, 3.75, 2.20, 0.36, 'Single Shared ConvNeXt-Tiny', 'ImageNet Pretrained (28M params)', bg='#eff6ff', border='#2563eb', bold=True)
C4  = box(2.20, 2.95, 0.50, 0.46, 'Stage C4', 's=4, 96ch\n(Frozen)')
C8  = box(2.78, 2.95, 0.50, 0.46, 'Stage C8', 's=8\n192ch')
C16 = box(3.36, 2.95, 0.50, 0.46, 'Stage C16', 's=16\n384ch')
C32 = box(3.94, 2.95, 0.50, 0.46, 'Stage C32', 's=32\n768ch')

# Bus from views into Shared Backbone
line(WV['r'], (1.78, WV['r'][1]), (1.78, C_BACK['l'][1]))
line(TTA['r'], (1.78, TTA['r'][1]), (1.78, C_BACK['l'][1]))
arrow((1.78, C_BACK['l'][1]), C_BACK['l'])

line(C_BACK['b'], (3.07, 3.35))
line((2.20, 3.35), (3.94, 3.35))
for st in [C4, C8, C16, C32]:
    arrow((st['t'][0], 3.35), st['t'])

# --- Stage 3 Content ---
PROJ = box(5.78, 3.75, 2.40, 0.36, 'Lateral 1x1 Projections', 'Maps C4, C8, C16, C32 -> 64 channels each')
FPN  = box(5.78, 2.95, 2.40, 0.48, 'Top-Down Additive Fusion', 'P32 -> P16 -> P8 -> P4 + 3x3 Smooth (GN + GELU)')
TLOG = box(5.78, 2.36, 2.40, 0.26, 'Dense Tile Logits M_tile', 'Bilinear Upsample to 384 x 384')

line(C4['r'], (4.38, C4['r'][1]))
line(C8['r'], (4.38, C8['r'][1]))
line(C16['r'], (4.38, C16['r'][1]))
line(C32['r'], (4.38, C32['r'][1]))
line((4.38, C4['r'][1]), (4.38, PROJ['l'][1]))
arrow((4.38, PROJ['l'][1]), PROJ['l'])

arrow(PROJ['b'], FPN['t'])
arrow(FPN['b'], TLOG['t'])

# --- Stage 4 Content ---
HANN = box(5.78, 1.55, 2.40, 0.42, '2D Separable Hann Window Weighting', 'w(u, v) with Positive Floor e_w = 0.05', bg='#f0fdf4', border='#16a34a')
HMAP = box(5.78, 0.85, 2.40, 0.44, 'Full-Canvas Anomaly Heatmap P(x)', 'Seamless Blending over Image Lattice (H x W)', bold=True)
BM   = box(5.78, 0.26, 2.40, 0.28, 'Binary Mask (t_mask = 0.35)', 'mIoU: 43.80% | Tiny IoU: 11.40%', bg='#fef9c3', border='#ca8a04')

arrow(TLOG['b'], HANN['t'])
arrow(HANN['b'], HMAP['t'])
arrow(HMAP['b'], BM['t'])

# --- Stage 5 Content ---
DH = box(3.55, 1.55, 1.30, 0.42, 'Global Image Head', 'GAP(C32) -> Linear -> s_g')
SH = box(2.05, 1.55, 1.30, 0.42, 'Local Tile Head', 'Top-100 Pixel Mean -> s_l')
HY = box(2.80, 0.95, 2.80, 0.32, 'Hybrid Score: s = 0.50 s_g + 0.50 s_l', 'Widened Separation Gap (4.3x Decompression)', bg='#f1f5f9', border='#0f172a', bold=True)

# Connection from C32 to Global Head
arrow(C32['b'], (3.94, 2.08), (3.55, 2.08), DH['t'])

# Connection from Full Map to Local Head
arrow(HMAP['l'], (4.38, HMAP['l'][1]), (4.38, 1.82), (2.05, 1.82), SH['t'])

arrow(DH['b'], (3.55, HY['t'][1]))
arrow(SH['b'], (2.05, HY['t'][1]))

# Verdict diamond and boxes
cx, cy = 1.45, 0.42
ax.add_patch(Polygon([(cx-0.42, cy), (cx, cy+0.20), (cx+0.42, cy), (cx, cy-0.20)], closed=True,
                     fc='#fee2e2', ec='#dc2626', lw=1.2, zorder=3))
ax.text(cx, cy, 't_d = 0.5200', ha='center', va='center', fontsize=5.8, fontweight='bold', color='#991b1b', zorder=4)

GEN = box(0.60, 0.42, 1.00, 0.42, 'AUTHENTIC', 's < 0.5200\n(85.2% Spec.)', bg='#f0fdf4', border='#16a34a')
EDT = box(2.45, 0.42, 1.15, 0.42, 'MANIPULATED', 's >= 0.5200\n(83.6% Acc.)', bg='#fef2f2', border='#ef4444')

arrow(HY['b'], (2.80, cy), (cx+0.42, cy))
arrow((cx-0.42, cy), GEN['r'])
arrow((cx+0.42, cy), EDT['l'])

# Save outputs to both Track B and root
out_dirs = [
    r'paper_work\FRAME_Tiny_Paper\figures',
    r'paper_work\manuscript\figures',
    r'C:\Users\DK11\.gemini\antigravity\brain\de528773-fc7f-4916-a14c-8040930e9d6a\visuals'
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)
    plt.savefig(os.path.join(d, 'pipeline_flowchart_halfpage.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(d, 'pipeline_flowchart_halfpage.png'), format='png', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(d, 'framework_flowchart.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(d, 'framework_flowchart.png'), format='png', dpi=300, bbox_inches='tight')

print('Successfully generated current architecture flowchart across all targets!')
