import os
import sys
import json
from pathlib import Path
import numpy as np
import torch

from baseline_adapter import VAASAdapter

def run_smoke_check():
    print("=== STARTING BASELINE ADAPTER SMOKE CHECK ===", flush=True)
    cfg = {
        "model_repo": "OBA-Research/vaas",
        "model_variant": "v2-base-df2023",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    print(f"Initializing VAASAdapter with config: {cfg}", flush=True)
    adapter = VAASAdapter(cfg)
    adapter.load_model()
    print("Model loaded successfully.", flush=True)
    
    test_img = Path("I:/icvgip workshop/research_data/opensdi_development_preview/001000_image.jpg")
    assert test_img.exists(), f"Missing test image: {test_img}"
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    print(f"Running inference on single image: {test_img}...", flush=True)
    pred = adapter.predict(str(test_img.resolve()))
    
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    print("Inference completed. Results:", flush=True)
    print("  Model ID:", pred["model_id"], flush=True)
    print("  Device:", pred["device"], flush=True)
    print("  Runtime:", pred["runtime_sec"], "seconds", flush=True)
    print("  Raw Scores:", pred["raw_scores"], flush=True)
    print("  Map Resolution:", pred["map_resolution"], flush=True)
    print("  Map Dtype:", pred["raw_map"].dtype, flush=True)
    print("  Map Range: min =", float(pred["raw_map"].min()), "max =", float(pred["raw_map"].max()), flush=True)
    print("  Map Finite (no NaN/Inf):", np.all(np.isfinite(pred["raw_map"])), flush=True)
    print(f"  Peak GPU Memory: {peak_mem_mb:.2f} MB", flush=True)
    
    # Assertions
    assert np.all(np.isfinite(pred["raw_map"])), "Error: Non-finite values in anomaly map!"
    assert pred["raw_map"].ndim == 2, f"Error: Map dimension is {pred['raw_map'].ndim}, expected 2"
    for k, v in pred["raw_scores"].items():
        assert np.isfinite(v), f"Error: Non-finite score for {k}: {v}"
        
    smoke_summary = {
        "status": "PASSED",
        "model_id": pred["model_id"],
        "device": pred["device"],
        "image_tested": str(test_img),
        "runtime_seconds": pred["runtime_sec"],
        "raw_scores": pred["raw_scores"],
        "map_resolution": pred["map_resolution"],
        "map_range": [float(pred["raw_map"].min()), float(pred["raw_map"].max())],
        "all_finite": True,
        "peak_gpu_memory_mb": round(peak_mem_mb, 2)
    }
    out_file = Path("I:/icvgip workshop/baseline_work/logs/smoke_check_log.json")
    out_file.write_text(json.dumps(smoke_summary, indent=2), encoding="utf-8")
    print(f"Saved smoke check log to {out_file}", flush=True)

if __name__ == "__main__":
    run_smoke_check()

