"""Find genuine control images with zero false alarms and low quiescent heatmaps using ConvNeXtTinyFRAMEDetector."""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label, binary_fill_holes

ROOT = Path("I:/icvgip workshop")
sys.path.insert(0, str(ROOT / "controlled_localization/src"))

from controlled_localization.models.convnext_tiny_frame import ConvNeXtTinyFRAMEDetector
from controlled_localization.train_and_evaluate_convnext_tiny import evaluate_model_on_image, remove_small_components

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_path = ROOT / "controlled_localization/runs/frame_convnext_tiny/best_convnext_tiny.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = ConvNeXtTinyFRAMEDetector(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    calibrated_tau = float(ckpt.get("opt_tau", 0.5200))
    print(f"Loaded ConvNeXt-Tiny model. tau_d = {calibrated_tau:.4f}")
    
    manifest_dir = ROOT / "baseline_work/data_manifests"
    generators = ["sdxl", "flux", "sd15", "sd2", "sd3"]
    
    candidates = []
    
    for gen in generators:
        m_file = manifest_dir / f"test_{gen}_manifest.json"
        if not m_file.exists():
            continue
        with open(m_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        entries = data.get("entries", [])
        clean_entries = [e for e in entries if int(e.get("label", 0)) == 0]
        print(f"Scanning {len(clean_entries)} clean entries for {gen}...")
        
        for e in clean_entries:
            img_rel = e.get("image_path")
            img_path = ROOT / img_rel
            if not img_path.exists():
                # try replacing backslashes
                img_path = ROOT / str(img_rel).replace("\\", "/")
            if not img_path.exists():
                continue
                
            with Image.open(img_path) as pil_img:
                rgb = pil_img.convert("RGB")
                w, h = rgb.size
                arr = np.array(rgb, dtype=np.float32) / 255.0
                
            sg, sl, sh, full_map = evaluate_model_on_image(
                model, arr, device, tile_size=384, stride=256, use_tta=True
            )
            
            # Post-processed mask thresholded at 0.35
            pred_bin = (full_map >= 0.35).astype(bool)
            pred_bin = binary_fill_holes(pred_bin)
            pred_bin = remove_small_components(pred_bin, min_size=64)
            fa_pixels = int(np.sum(pred_bin))
            fa_rate = float(fa_pixels) / float(w * h)
            
            max_prob = float(np.max(full_map))
            mean_prob = float(np.mean(full_map))
            
            rec = {
                "gen": gen,
                "sample_id": e.get("sample_id", Path(img_rel).stem),
                "image_path": str(img_path).replace("\\", "/"),
                "w": w,
                "h": h,
                "s_global": sg,
                "s_local": sl,
                "s_hybrid": sh,
                "max_prob": max_prob,
                "mean_prob": mean_prob,
                "fa_pixels": fa_pixels,
                "fa_rate": fa_rate
            }
            candidates.append(rec)
            
            if fa_pixels == 0 and max_prob < 0.25:
                print(f"  [0 FA] {e.get('sample_id')} | s={sh:.4f} | max_P={max_prob:.4f} | mean_P={mean_prob:.4f}")
                    
    # Sort candidates by zero false alarms, then lowest max_prob, then lowest s_hybrid
    zero_fa = [c for c in candidates if c["fa_pixels"] == 0]
    zero_fa.sort(key=lambda x: (x["max_prob"], x["s_hybrid"]))
    
    print(f"\nTotal scanned: {len(candidates)}, with 0 false alarms: {len(zero_fa)}")
    print("\nTop 10 candidates with 0.00% False Alarms:")
    for i, c in enumerate(zero_fa[:10]):
        print(f"[{i+1}] {c['sample_id']} | s={c['s_hybrid']:.4f} | max_P={c['max_prob']:.4f} | mean_P={c['mean_prob']:.4f} | {c['image_path']}")
        
    # Save results to json
    with open(ROOT / "scripts/clean_candidates.json", "w", encoding="utf-8") as f:
        json.dump(zero_fa, f, indent=2)

if __name__ == "__main__":
    main()
