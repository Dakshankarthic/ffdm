import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
import torch

class BaselineAdapter:
    """Base interface for all forensic baseline runners."""
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None

    def load_model(self):
        raise NotImplementedError

    def predict(self, image_path: str) -> dict:
        """
        Receives ONE single image path.
        No target masks or ground truth are passed.
        Returns:
            {
                "raw_scores": dict of float scores,
                "raw_map": np.ndarray (float32, 2D),
                "map_resolution": (H, W),
                "runtime_sec": float,
                "device": str,
                "model_id": str
            }
        """
        raise NotImplementedError

class VAASAdapter(BaselineAdapter):
    def load_model(self):
        os.environ['HF_HOME'] = str(Path('I:/icvgip workshop/baselines/model_cache').resolve())
        from vaas.inference.pipeline import VAASPipeline
        repo_id = self.config.get("model_repo", "OBA-Research/vaas")
        variant = self.config.get("model_variant", "v2-base-df2023")
        device_str = "cuda" if (self.device.type == "cuda" and torch.cuda.is_available()) else "cpu"
        self.pipeline = VAASPipeline.from_pretrained(
            repo_id=repo_id,
            device=device_str,
            model_variant=variant
        )
        self.model_id = f"{repo_id}:{variant}"

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        t0 = time.perf_counter()
        # forward_detailed returns S_F, S_P, S_H, anomaly_map
        res = self.pipeline.forward_detailed(image_path)
        runtime = time.perf_counter() - t0
        
        raw_map = np.asarray(res["anomaly_map"], dtype=np.float32)
        
        return {
            "raw_scores": {
                "S_F": float(res["S_F"]),
                "S_P": float(res["S_P"]),
                "S_H": float(res["S_H"]),
            },
            "raw_map": raw_map,
            "map_resolution": list(raw_map.shape),
            "runtime_sec": round(runtime, 4),
            "device": str(self.device),
            "model_id": self.model_id,
        }

