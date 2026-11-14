"""Multi-Evidence Fusion Head for Whole-Image + Native-Resolution Tiled Classification.
Combines global context from resized whole images with high-resolution localized forensic evidence
from native tiles (verified edit regions + hard negative textured genuine areas).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiEvidenceDetectionHead(nn.Module):
    """
    Combines:
    - e_whole (257-dim: gap32 [128] + gmp32 [128] + topk_mean [1])
    - e_tiles (N x 257-dim for sampled native-resolution tiles)
    Produces:
    - whole_logit [B, 1]
    - tile_logits [B, N, 1]
    - combined_logit [B, 1]
    """

    def __init__(self, in_dim: int = 257, hidden_dim: int = 64) -> None:
        super().__init__()

        # Whole-image classification path
        self.whole_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Tile-level classification path (shared across tiles)
        self.tile_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Multi-evidence fusion combiner
        # Inputs: whole_logit [1], max(tile_logit) [1], mean(tile_logit) [1], whole_topk [1], max(tile_topk) [1]
        self.combiner = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        e_whole: torch.Tensor,               # [B, 257]
        e_tiles: torch.Tensor,               # [B, N, 257]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          combined_logit: [B, 1]
          whole_logit: [B, 1]
          tile_logits: [B, N, 1]
        """
        b, n, d = e_tiles.shape

        # 1. Whole image forward
        whole_logit = self.whole_mlp(e_whole)  # [B, 1]
        whole_topk = e_whole[:, -1:]           # [B, 1]

        # 2. Tiles forward
        e_tiles_flat = e_tiles.view(b * n, d)
        tile_logits_flat = self.tile_mlp(e_tiles_flat)
        tile_logits = tile_logits_flat.view(b, n, 1)

        tile_topk = e_tiles[:, :, -1:]  # [B, N, 1]
        max_tile_topk, _ = torch.max(tile_topk, dim=1)  # [B, 1]

        max_tile_logit, _ = torch.max(tile_logits, dim=1)    # [B, 1]
        mean_tile_logit = torch.mean(tile_logits, dim=1)     # [B, 1]

        # 3. Multi-Evidence Combiner
        fusion_input = torch.cat([
            whole_logit,
            max_tile_logit,
            mean_tile_logit,
            whole_topk,
            max_tile_topk,
        ], dim=-1)  # [B, 5]

        combined_logit = self.combiner(fusion_input)

        return combined_logit, whole_logit, tile_logits
