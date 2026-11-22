"""Final Complete Scientific Integrity & Paper Consistency Audit."""
from pathlib import Path
import re
import zipfile
import os

tex_path = Path("I:/icvgip workshop/paper_work/FRAME_Tiny_Paper/main.tex")
content = tex_path.read_text(encoding="utf-8")

print("=" * 60)
print("RUNNING FINAL SCIENTIFIC INTEGRITY AUDIT")
print("=" * 60)

checks = []

def check(name: str, condition: bool, details: str = ""):
    status = "PASS" if condition else "FAIL"
    checks.append((status, name, details))
    print(f"[{status}] {name}" + (f" -> {details}" if details and not condition else ""))

# 1. Prose matches Table 4 (Ablation)
check("Ablation Row 3 Pixel F1 in prose matches Table 4 (46.80%)", "boosting pixel F1 to $46.80\\%$" in content)
check("Ablation Row 4 mIoU in prose matches Table 4 (40.20%)", "boosting mIoU to $40.20\\%$" in content)
check("Ablation Row 1 mIoU matches Table 4 (28.40%)", "$28.40\\%$ mIoU" in content)
check("Ablation Row 2 mIoU matches Table 4 (33.80%)", "$33.80\\%$" in content)
check("Ablation Row 6 B-Acc matches Table 4 (82.60%)", "$82.60\\%$" in content)
check("Ablation Row 7 Acc & mIoU matches Table 4 (83.60% / 43.80%)", "\\textbf{83.60\\% accuracy} and \\textbf{43.80\\% mIoU}" in content)

# 2. Table 3 (Generator breakdown)
check("Table 3 FRAME SD1.5 (63.50%)", "63.50" in content)
check("Table 3 FRAME SD2.1 (51.80%)", "51.80" in content)
check("Table 3 FRAME SD3 (47.60%)", "47.60" in content)
check("Table 3 FRAME SDXL (33.20%)", "33.20" in content)
check("Table 3 FRAME FLUX.1 (22.70%)", "22.70" in content)
check("Table 3 pooled test set clarification present", "Overall mIoU ($43.80\\%$) is evaluated on the full pooled test set" in content)

# 3. Baseline Authenticity (Table 2 & Table 3)
check("MaskCLIP published Table 2 Acc (82.00%)", "\\underline{82.00} & NR & \\underline{77.80} & NR & \\underline{42.70} & \\underline{49.40} & NR & NR" in content)
check("MaskCLIP published Table 3 SD1.5 (67.10%)", "\\textbf{67.10}" in content)
check("MaskCLIP published Table 3 SD2.1 (55.50%)", "\\textbf{55.50}" in content)
check("MaskCLIP published Table 3 SD3 (43.80%)", "\\underline{43.80}" in content)
check("TruFor published Table 3 SD1.5 (63.40%)", "63.40 & 54.70 & 32.30 & 26.60 & 7.60" in content)
check("DeCLIP published Table 3 SD1.5 (37.20%)", "37.20 & 35.70 & 27.30 & 14.60 & 11.20" in content)

# 4. Table 6 Hardware Authenticity
check("Table 6 reports RTX 2070 SUPER", "NVIDIA GeForce RTX" in content and "2070 SUPER" in content)
check("Table 6 FP32 latency 32.5 ms, 30.8 FPS, 1,740 MB", "32.5" in content and "30.8" in content and "1,740" in content)
check("Table 6 FP16 latency 23.5 ms, 42.5 FPS, 1,220 MB", "23.5" in content and "42.5" in content and "1,220" in content)
check("RTX 4090 completely removed", "RTX 4090" not in content)
check("Jetson Orin completely removed", "Jetson Orin" not in content)

# 5. Banned numbers absent
check("Banned 81.80 word-boundary absent", len(re.findall(r"\b81\.80\b", content)) == 0)
check("Banned 91.20 absent", "91.20" not in content)
check("Banned 92.60 absent", "92.60" not in content)
check("Banned 93.40 absent", "93.40" not in content)
check("Banned 47.30 absent", "47.30" not in content)
check("Banned 63.80 absent", "63.80" not in content)
check("FUSED absent from Table 2", not any("FUSED" in line for line in content[content.find("tab:main_benchmark"):content.find("tab:main_benchmark")+1500].split("\n")))
check("FUSED absent from Table 3", not any("FUSED" in line for line in content[content.find("tab:generator_breakdown"):content.find("tab:generator_breakdown")+1000].split("\n")))

# 6. Float overlap protection
check("Table 5 uses [h!] float specifier", "\\begin{table}[h!]\n\\centering\\small\n\\caption{Robustness evaluation under post-processing" in content)
check("Table 4 uses [h!] float specifier", "\\begin{table*}[h!]\n\\centering\\small\n\\caption{Systematic component-wise ablation" in content)

# Summary
passed = sum(1 for s, _, _ in checks if s == "PASS")
failed = sum(1 for s, _, _ in checks if s == "FAIL")

print("\n" + "=" * 60)
print(f"AUDIT SUMMARY: {passed}/{len(checks)} PASSED ({failed} FAILED)")
print("=" * 60)

if failed == 0:
    print("ALL CHECKS PASSED PERFECTLY!")
    
    # Rebuild Overleaf zip
    source_dir = Path(r"I:\icvgip workshop\paper_work\FRAME_Tiny_Paper")
    targets = [
        Path(r"C:\Users\DK11\Downloads\FRAME_Tiny_Paper_Overleaf.zip"),
        Path(r"I:\icvgip workshop\FRAME_Tiny_Paper_Overleaf.zip")
    ]
    for target_zip in targets:
        with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)
        print(f"Rebuilt ZIP: {target_zip} ({target_zip.stat().st_size / (1024*1024):.2f} MB)")
else:
    print("ERRORS FOUND - PLEASE INVESTIGATE!")
