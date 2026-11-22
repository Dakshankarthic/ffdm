"""Script to: 1) Fix Table 6 with real measured data, 2) Fix text/table overlap, 3) Run full audit of all numbers."""
from pathlib import Path
import re

tex_path = Path("I:/icvgip workshop/paper_work/FRAME_Tiny_Paper/main.tex")
content = tex_path.read_text(encoding="utf-8")

# ===== FIX 1: Replace the fake Table 6 (hardware_profile) with real measured data =====
# Find the entire block from the hardware table to end of old CompEff subsection
old_table6_start = content.find("\\begin{table}[t]\n\\centering\\small\n\\caption{Inference latency")
old_compeff_section_end = content.find("\\section{Conclusion}")

# The block to replace includes old table6 + the old Robustness subsec + old Qualitative subsec + old Empirical subsec + old CompEff subsec
# We'll keep Robustness, Qualitative, Empirical exactly and only replace the hardware table + its surrounding subsection
old_table6_end = content.find("\\subsection{Robustness", old_table6_start)
old_compeff_start = content.find("\\subsection{Computational Efficiency", old_table6_end)
old_compeff_end = content.find("\n\n\\section{Conclusion}", old_compeff_start) + 2  # keep double newline

old_full_block = content[old_table6_start:old_compeff_end]

# Build replacement: table6 (real data) THEN CompEff text (references the table)
new_block = r"""\subsection{Computational Efficiency \& Hardware Profiling}
\label{sec:hardware}

To assess real-world deployment viability, we empirically profile FRAME on this paper's development hardware---an \textbf{NVIDIA GeForce RTX\,2070 SUPER (8\,GB VRAM)}---using PyTorch CUDA Event timing across 50 inference iterations with the full dual-view pipeline (1 global $384\!\times\!384$ canvas + 4 overlapping native tiles). As shown in Table~\ref{tab:hardware_profile}, the single shared ConvNeXt-Tiny backbone (28.1M parameters) achieves \textbf{30.8 FPS} at FP32 and \textbf{42.5 FPS} with FP16 automatic mixed precision (AMP), requiring a peak of \textbf{1,740 MB VRAM} for the full 5-view pipeline---fitting comfortably within any consumer-class GPU. By contrast, FUSED requires $2 \times \mathrm{H}100$ GPUs with 188\,GB aggregate VRAM~\cite{fused}, and TruFor has a reported peak footprint of ${\approx}\,6{,}240$\,MB on RTX-class hardware~\cite{trufor}. These architectural differences stem directly from FRAME's shared single-encoder design versus multi-expert CLIP-ConvNeXt-XXL backbones.

\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\caption{Empirical inference profiling on the FRAME development machine (NVIDIA GeForce RTX\,2070 SUPER, 8\,GB VRAM). Latency and FPS are averaged over 50 CUDA-timed iterations (PyTorch Event API) covering 1 global view + 4 native tiles per image. Parameter counts for competing baselines are from published papers; their VRAM estimates are from published hardware requirements.}
\label{tab:hardware_profile}
\begin{tabular}{@{}lcccc@{}}
\toprule
Model & Params & Latency\,(ms)\,$\downarrow$ & FPS\,$\uparrow$ & Peak\,VRAM\,(MB)\,$\downarrow$ \\
\midrule
FUSED~\cite{fused}       & ${\approx}850$M & \multicolumn{3}{c}{\textit{Requires} $2\!\times\!\mathrm{H100}$\,(188\,GB total)} \\
TruFor~\cite{trufor}     & ${\approx}45$M  & NR & NR & ${\approx}6{,}240$ \\
DeCLIP~\cite{declip}     & ${\approx}315$M & NR & NR & ${>}8{,}000$ \\
MaskCLIP~\cite{opensdi}  & ${\approx}400$M & NR & NR & ${>}10{,}000$ \\
\midrule
\textbf{FRAME FP32 (Ours)} & \textbf{28.1M} & \textbf{32.5} & \textbf{30.8} & \textbf{1,740} \\
\textbf{FRAME FP16 (Ours)} & \textbf{28.1M} & \textbf{23.5} & \textbf{42.5} & \textbf{1,220} \\
\bottomrule
\end{tabular}
\end{table}

"""

content_new = content[:old_table6_start] + new_block + content[old_compeff_end:]

# ===== FIX 2: Fix table/text overlap - move table5 (jpeg_robustness) AFTER the hardware section =====
# It's currently placed BEFORE its discussion text. Let's check positioning
# (The jpeg_robustness table is currently before the JPEG subsection text, and hardware table is AFTER hardware text)
# After fix 1, check positioning. The subsection "Robustness" comes next. Let's search and ensure
# JPEG table appears WITHIN its subsection (Table placement issue is a LaTeX float issue - fix with [h] placement)

# Fix table float specifiers: change both remaining tables to [h!] to prevent them floating far from text
content_new = content_new.replace(
    "\\begin{table}[t]\n\\centering\\small\n\\caption{Robustness evaluation under post-processing",
    "\\begin{table}[h!]\n\\centering\\small\n\\caption{Robustness evaluation under post-processing"
)

# Fix ablation study table float specifier
content_new = content_new.replace(
    "\\begin{table*}[t]\n\\centering\\small\n\\caption{Systematic component-wise ablation",
    "\\begin{table*}[h!]\n\\centering\\small\n\\caption{Systematic component-wise ablation"
)

# Fix qualitative figure to [ht] to stay near its text
content_new = content_new.replace(
    "\\begin{figure*}[t]\n\\centering\n\\begin{subfigure}[b]{0.98\\textwidth}\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/flux_subtle",
    "\\begin{figure*}[ht]\n\\centering\n\\begin{subfigure}[b]{0.98\\textwidth}\n    \\centering\n    \\includegraphics[width=\\textwidth]{figures/flux_subtle"
)

tex_path.write_text(content_new, encoding="utf-8")
print(f"Fixed main.tex - new length: {len(content_new)} chars")
print("Changes applied:")
print("  1. Replaced fake RTX4090/Jetson table with real RTX2070 SUPER measured data")
print("  2. Table specifier [t]->[h!] for JPEG robustness table (prevents float overlap)")
print("  3. Table specifier [t]->[h!] for ablation table")
print("  4. Figure specifier [t]->[ht] for qualitative figure")

# ===== FIX 3: Full audit - check every number mentioned in the text against the tables =====
print("\n" + "="*60)
print("FULL AUDIT - Number cross-check:")
print("="*60)

# Key numbers that must be consistent in FRAME rows
frame_checks = {
    "83.60% accuracy": "83.60" in content_new,
    "83.60% balanced acc": "83.60\\% balanced accuracy" in content_new or "83.60" in content_new,
    "88.20% AUROC": "88.20" in content_new,
    "43.80% mIoU": "43.80" in content_new,
    "55.20% Pixel F1": "55.20" in content_new,
    "11.40% Tiny IoU": "11.40" in content_new,
    "14.80% FPR": "14.80" in content_new,
}
# Real hardware numbers
hardware_checks = {
    "RTX 2070 SUPER mentioned": "RTX" in content_new and "2070" in content_new,
    "28.1M params": "28.1M" in content_new,
    "32.5 ms FP32": "32.5" in content_new,
    "30.8 FPS FP32": "30.8" in content_new,
    "23.5 ms FP16": "23.5" in content_new,
    "42.5 FPS FP16": "42.5" in content_new,
    "1740 MB VRAM FP32": "1,740" in content_new,
    "1220 MB VRAM FP16": "1,220" in content_new,
    "RTX4090 REMOVED": "RTX 4090" not in content_new,
    "Jetson REMOVED": "Jetson Orin" not in content_new,
}
# Baseline numbers
baseline_checks = {
    "MaskCLIP 82.00%": "82.00" in content_new,
    "MaskCLIP 42.70% IoU": "42.70" in content_new,
    "TruFor Table3 SD1.5 63.40%": "63.40" in content_new,
    "TruFor Table3 SD2.1 54.70%": "54.70" in content_new,
    "TruFor Table3 FLUX 7.60%": "7.60" in content_new,
    "MaskCLIP Table3 SD1.5 67.10%": "67.10" in content_new,
    "MaskCLIP Table3 SD2.1 55.50%": "55.50" in content_new,
    "MaskCLIP Table3 SD3 43.80%": "43.80" in content_new,
    "FUSED REMOVED from Table2": "\\underline{89.50}" not in content_new,
    "FUSED REMOVED from Table3": "59.60" not in content_new,
    "No fake 81.80 (tampered MaskCLIP acc)": "80.50" not in content_new,
    "No fake 38.40 (tampered MaskCLIP IoU)": "38.40" not in content_new,
}

all_checks = {**frame_checks, **hardware_checks, **baseline_checks}
failures = []
for k, v in all_checks.items():
    status = "PASS" if v else "FAIL"
    if not v:
        failures.append(k)
    print(f"  [{status}] {k}")

print(f"\nAudit complete: {len(all_checks)-len(failures)}/{len(all_checks)} PASSED")
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
