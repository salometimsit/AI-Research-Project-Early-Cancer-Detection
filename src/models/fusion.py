"""Cross-attention fusion of radiomics + SwinViT deep features.

Pipeline stage: L in the A->M flow.

Inputs
------
* ``radiomics_dim`` -- length of the VaRFS-stable feature vector.
* ``deep_dim`` -- length of the SwinViT deep feature vector.
* ``cross_attention.num_heads`` and ``cross_attention.hidden_dim``
  from the config.

Outputs
-------
* ``forward(radiomics_vec, deep_vec)`` -> fused tensor of shape
  ``(B, output_dim)`` ready for the classification head.

Failure modes
-------------
* Mismatched batch dims between the two streams -> standard
  ``MultiheadAttention`` errors.
* Wrong projection dims -> shape mismatch raised at construction time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Joint embedding via multi-head cross attention.

    The radiomics vector queries the SwinViT deep vector (and vice-versa),
    yielding a fused representation that captures correlations between
    classical texture features and learned visual features.
    """

    def __init__(
        self,
        radiomics_dim: int,
        deep_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.radiomics_proj = nn.Linear(radiomics_dim, hidden_dim)
        self.deep_proj = nn.Linear(deep_dim, hidden_dim)
        self.norm_radio_in = nn.LayerNorm(hidden_dim)
        self.norm_deep_in = nn.LayerNorm(hidden_dim)
        self.attn_radio_to_deep = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_deep_to_radio = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.pre_mlp_dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(
        self,
        radiomics_features: torch.Tensor,
        deep_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse two ``(B, dim)`` vectors into a single ``(B, hidden_dim)`` vector."""
        radio = self.norm_radio_in(self.radiomics_proj(radiomics_features)).unsqueeze(1)
        deep = self.norm_deep_in(self.deep_proj(deep_features)).unsqueeze(1)

        # Parallel cross-attention from the same clean states (no serial bias).
        radio_attn, _ = self.attn_radio_to_deep(radio, deep, deep)
        deep_attn, _ = self.attn_deep_to_radio(deep, radio, radio)
        radio = self.norm1(radio + radio_attn)
        deep = self.norm2(deep + deep_attn)

        combined = torch.cat([radio.squeeze(1), deep.squeeze(1)], dim=-1)
        combined = self.pre_mlp_dropout(combined)
        return self.mlp(combined)
