"""Phase 4: Reinforcement Learning (RL) Sequential Tile Inspection Pilot.
Formulates sequential tile inspection under a compute budget constraint:
  - State: Whole-image embedding, inspection history mask, step count, max tile score.
  - Action Space: 9 candidate grid tiles + TERMINATE action.
  - Budget: Maximum K=4 tile inspections.
  - Reward: Supervised training reward penalizing false alarms & step count, rewarding IoU & accuracy.
Fairly benchmarks:
  1. Fixed Grid Selection
  2. Uniform Random Selection
  3. Whole-Image Heuristic Selection
  4. Trained Sequential RL Policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from controlled_localization.models.super_frame_v2_detector import SuperFRAMEV2Detector
from controlled_localization.data.tiles import get_tile_1d_starts, blend_tiles_probability
from controlled_localization.utils import seed_all


ROOT = Path("I:/icvgip workshop")
RUNS_DIR = ROOT / "controlled_localization/runs"
CKPT_PATH = RUNS_DIR / "scaled_pilot/best_superframe_v2_scaled.pt"
DEV_MANIFEST = RUNS_DIR / "real_development_20261114/manifest_multigen.json"


class PolicyNetwork(nn.Module):
    """Lightweight 2-layer MLP policy network mapping inspection state to action logits."""

    def __init__(self, state_dim: int = 16, hidden_dim: int = 64, num_actions: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> Categorical:
        # mask is [B, num_actions], with 1 for valid actions and 0 for invalid (already inspected)
        logits = self.net(state)
        # Large negative value for invalid actions
        invalid = (mask == 0)
        logits[invalid] = -1e9
        return Categorical(logits=logits)


def get_3x3_grid_coords(width: int, height: int, tile_size: int = 384) -> List[Tuple[int, int]]:
    """Returns 9 tile coordinates covering the 3x3 spatial grid."""
    x_starts = get_tile_1d_starts(width, tile_size=tile_size, stride=(width - tile_size) // 2 if width > tile_size else 1)
    y_starts = get_tile_1d_starts(height, tile_size=tile_size, stride=(height - tile_size) // 2 if height > tile_size else 1)
    # Ensure exactly 3 in each dimension
    if len(x_starts) < 3:
        x_starts = [0, max(0, (width - tile_size) // 2), max(0, width - tile_size)]
    elif len(x_starts) > 3:
        x_starts = [x_starts[0], x_starts[len(x_starts) // 2], x_starts[-1]]

    if len(y_starts) < 3:
        y_starts = [0, max(0, (height - tile_size) // 2), max(0, height - tile_size)]
    elif len(y_starts) > 3:
        y_starts = [y_starts[0], y_starts[len(y_starts) // 2], y_starts[-1]]

    coords = []
    for y in y_starts[:3]:
        for x in x_starts[:3]:
            coords.append((x, y))
    return coords[:9]


def extract_tile(arr_full: np.ndarray, x: int, y: int, tile_size: int = 384) -> torch.Tensor:
    h, w = arr_full.shape[:2]
    tx_end = min(w, x + tile_size)
    ty_end = min(h, y + tile_size)
    patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
    sub = arr_full[y:ty_end, x:tx_end]
    patch[: sub.shape[0], : sub.shape[1]] = sub
    return torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()


def run_episode(
    model: nn.Module,
    policy: nn.Module | None,
    mode: str,  # 'rl', 'grid', 'random', 'heuristic'
    img_arr: np.ndarray,
    gt_mask: np.ndarray,
    label: float,
    device: torch.device,
    max_budget: int = 4,
    tile_size: int = 384,
) -> Tuple[float, float, int, float, List[torch.Tensor]]:
    """
    Runs a sequential inspection episode on one image.
    Returns:
      final_score: float
      iou: float
      steps_taken: int
      reward: float
      log_probs: list of action log probabilities
    """
    h, w = img_arr.shape[:2]
    grid_coords = get_3x3_grid_coords(w, h, tile_size=tile_size)

    # 1. Whole-image view
    whole_resized = Image.fromarray((img_arr * 255).astype(np.uint8)).resize((tile_size, tile_size), Image.BILINEAR)
    whole_arr = np.array(whole_resized, dtype=np.float32) / 255.0
    whole_t = torch.from_numpy(whole_arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.amp.autocast("cuda"):
        whole_out = model(whole_t)
    whole_logit = whole_out["det_logits"][0, 0].item()
    s_whole = float(torch.sigmoid(torch.tensor(whole_logit)).item())
    whole_mask_prob = torch.sigmoid(whole_out["mask_logits"][0, 0]).cpu().numpy()

    # Pre-rank tiles for heuristic mode
    if mode == "heuristic":
        # Rank grid coordinates by mean activation in whole-image map
        grid_ranks = []
        for idx, (gx, gy) in enumerate(grid_coords):
            # map gx, gy to 384x384 coordinates
            mx1 = int(gx / w * tile_size)
            my1 = int(gy / h * tile_size)
            mx2 = int(min(w, gx + tile_size) / w * tile_size)
            my2 = int(min(h, gy + tile_size) / h * tile_size)
            sub_m = whole_mask_prob[my1:my2, mx1:mx2]
            score = float(np.mean(sub_m)) if sub_m.size > 0 else 0.0
            grid_ranks.append((score, idx))
        heuristic_order = [idx for _, idx in sorted(grid_ranks, reverse=True)]

    inspected_mask = [0] * 9
    inspected_coords = []
    tile_logits = []
    log_probs = []

    steps_taken = 0
    s_max_tile = 0.0

    for step in range(max_budget):
        steps_taken += 1

        # Determine action
        if mode == "rl" and policy is not None:
            # Construct state vector: [s_whole, s_max_tile, step / max_budget, inspected_mask (9)]
            state_vec = torch.tensor([s_whole, s_max_tile, step / max_budget] + inspected_mask + [0.0] * 4, dtype=torch.float32, device=device).unsqueeze(0)
            valid_actions = [1 - m for m in inspected_mask] + [1]  # 9 tiles + 1 terminate
            valid_tensor = torch.tensor(valid_actions, dtype=torch.float32, device=device).unsqueeze(0)

            dist = policy(state_vec, valid_tensor)
            action_idx = dist.sample()
            log_probs.append(dist.log_prob(action_idx))
            action = action_idx.item()

            if action == 9:  # TERMINATE early
                break
        elif mode == "grid":
            # Fixed grid: e.g. center (4), top-left (0), bottom-right (8), top-right (2)
            fixed_seq = [4, 0, 8, 2]
            action = fixed_seq[step]
        elif mode == "random":
            available = [i for i, m in enumerate(inspected_mask) if m == 0]
            action = random.choice(available)
        elif mode == "heuristic":
            action = heuristic_order[step]
        else:
            action = 0

        # Execute inspection of chosen tile
        inspected_mask[action] = 1
        gx, gy = grid_coords[action]
        inspected_coords.append((gx, gy))

        tile_t = extract_tile(img_arr, gx, gy, tile_size=tile_size).to(device)
        with torch.amp.autocast("cuda"):
            t_out = model(tile_t)
        t_mask_l = t_out["mask_logits"][0, 0].cpu().numpy()
        tile_logits.append(t_mask_l)

        tile_anomaly = float(np.mean(np.sort(t_mask_l.ravel())[-50:]))
        if tile_anomaly > s_max_tile:
            s_max_tile = tile_anomaly

    # Blend inspected tiles
    blended_map = blend_tiles_probability(tile_logits, inspected_coords, w, h, tile_size=tile_size)
    s_pool = float(np.mean(np.sort(blended_map.ravel())[-100:]))
    final_score = 0.15 * s_whole + 0.85 * s_pool

    pred_bin = (blended_map > 0.35).astype(np.float32)

    # Compute IoU & Reward
    if label == 1.0:
        inter = np.sum(pred_bin * gt_mask)
        union = np.sum((pred_bin + gt_mask) > 0)
        iou = float(inter / union) if union > 0 else 0.0
        # Edited reward: +1.0 for high score, +2.0 * IoU, -0.05 * steps
        r_det = 1.0 if final_score >= 0.35 else -1.5
        reward = r_det + 2.0 * iou - 0.05 * steps_taken
    else:
        fp_rate = float(np.mean(pred_bin > 0))
        iou = 0.0
        # Genuine reward: +1.0 if correct, -1.5 if false alarm, -0.05 * steps
        r_det = 1.0 if final_score < 0.35 else -1.5
        reward = r_det - 2.0 * fp_rate - 0.05 * steps_taken

    return final_score, iou, steps_taken, reward, log_probs


def train_and_evaluate_rl(
    train_samples: int = 200,
    val_samples: int = 100,
    episodes: int = 150,
    seed: int = 42,
) -> Dict:
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"PHASE 4: RL SEQUENTIAL TILE INSPECTION BENCHMARK ({device})")
    print(f"=======================================================")

    # Load frozen SuperFRAME detector
    model = SuperFRAMEV2Detector(pretrained=False).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load data
    with open(DEV_MANIFEST, "r", encoding="utf-8") as f:
        all_recs = json.load(f)

    train_recs = [r for r in all_recs if r["split"] == "train"][:train_samples]
    val_recs = [r for r in all_recs if r["split"] == "val"][:val_samples]

    # Preload image arrays & masks into memory for lightning-fast training
    print(f"Preloading {len(train_recs)} training and {len(val_recs)} validation images...")
    def load_cache(recs):
        cache = []
        for r in recs:
            with Image.open(r["image"]) as img:
                arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
            lbl = float(r["label"])
            if lbl == 1.0 and r.get("mask") and Path(r["mask"]).exists():
                with Image.open(r["mask"]) as m_img:
                    gt = (np.array(m_img.convert("L"), dtype=np.float32) > 127.5).astype(np.float32)
            else:
                gt = np.zeros(arr.shape[:2], dtype=np.float32)
            cache.append((arr, gt, lbl))
        return cache

    train_cache = load_cache(train_recs)
    val_cache = load_cache(val_recs)

    # Instantiate Policy Network
    policy = PolicyNetwork(state_dim=16, hidden_dim=64, num_actions=10).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    print(f"Training Policy Network via REINFORCE for {episodes} episodes...")
    t0 = time.time()
    moving_baseline = 0.0

    for ep in range(1, episodes + 1):
        # Sample random training image
        arr, gt, lbl = random.choice(train_cache)
        _, _, _, reward, log_probs = run_episode(
            model, policy, mode="rl", img_arr=arr, gt_mask=gt, label=lbl, device=device, max_budget=4
        )

        moving_baseline = 0.95 * moving_baseline + 0.05 * reward
        advantage = reward - moving_baseline

        if log_probs:
            loss = -torch.stack(log_probs).sum() * advantage
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        if ep % 50 == 0 or ep == episodes:
            print(f"[RL Training] Episode {ep:3d}/{episodes} | Reward: {reward:.3f} | Baseline: {moving_baseline:.3f} | Elapsed: {(time.time()-t0):.1f}s")

    # Benchmarking on Validation Set
    print("\nRunning Benchmark on Frozen Validation Set...")
    methods = ["grid", "random", "heuristic", "rl"]
    benchmark_results = {}

    for m in methods:
        t_start = time.time()
        scores = []
        ious = []
        steps = []
        labels = []

        for arr, gt, lbl in val_cache:
            p_net = policy if m == "rl" else None
            score, iou, step_cnt, _, _ = run_episode(
                model, p_net, mode=m, img_arr=arr, gt_mask=gt, label=lbl, device=device, max_budget=4
            )
            scores.append(score)
            labels.append(lbl)
            if lbl == 1.0:
                ious.append(iou)
            steps.append(step_cnt)

        elapsed_ms = (time.time() - t_start) / len(val_cache) * 1000

        try:
            auroc = float(roc_auc_score(labels, scores))
        except Exception:
            auroc = 0.5

        # Best accuracy
        best_acc = 0.0
        best_f1 = 0.0
        for th in np.linspace(0.1, 0.9, 33):
            preds = [1 if s >= th else 0 for s in scores]
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            if acc > best_acc:
                best_acc = acc
                best_f1 = f1

        mean_iou = float(np.mean(ious)) if ious else 0.0
        avg_steps = float(np.mean(steps))

        benchmark_results[m] = {
            "mean_iou": round(mean_iou * 100, 2),
            "accuracy": round(best_acc * 100, 2),
            "f1": round(best_f1 * 100, 2),
            "auroc": round(auroc * 100, 2),
            "avg_tiles_inspected": round(avg_steps, 2),
            "latency_ms_per_image": round(elapsed_ms, 1),
        }

    out_p = ROOT / "controlled_localization/reports/phase4_rl_pilot_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S IST"),
            "seed": seed,
            "max_budget": 4,
            "val_samples": len(val_cache),
            "results": benchmark_results,
        }, f, indent=2)

    print("\n=======================================================")
    print("PHASE 4 RL TILE INSPECTION BENCHMARK RESULTS:")
    print("=======================================================")
    print("Method            | Accuracy | F1 Score | AUROC  | Mean IoU | Avg Tiles | Latency (ms)")
    print("------------------|----------|----------|--------|----------|-----------|-------------")
    for m, res in benchmark_results.items():
        name = "Uniform Grid" if m == "grid" else ("Random Tiles" if m == "random" else ("Heuristic" if m == "heuristic" else "RL Policy (Ours)"))
        print(f"{name:<17} | {res['accuracy']:>7.2f}% | {res['f1']:>7.2f}% | {res['auroc']:>5.2f}% | {res['mean_iou']:>7.2f}% | {res['avg_tiles_inspected']:>9.2f} | {res['latency_ms_per_image']:>11.1f}")
    print(f"\nSaved benchmark results to: {out_p}")
    return benchmark_results


if __name__ == "__main__":
    train_and_evaluate_rl(train_samples=150, val_samples=80, episodes=100)
