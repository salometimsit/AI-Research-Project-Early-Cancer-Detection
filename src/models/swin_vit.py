"""3D SwinViT feature extractor built on MONAI.

Pipeline stage: J + K in the A->M flow.

Inputs
------
* Cropped liver volumes ``(B, 1, D, H, W)`` produced by
  :class:`src.data.dataset.HCCDataset`.
* ``swin_vit.*`` config block (``img_size``, ``patch_size``,
  ``embed_dim``, ``depths``, ``num_heads``, ``dropout``).
* Optional ``swin_vit.pretrained_weights`` — backbone warm-start applied in
  :mod:`src.models.cv_training` via :mod:`src.models.pretrained_swin`, not in ``__init__``.

Outputs
-------
* ``forward(x)`` -> ``(logits, deep_features)``.
* ``extract_features(x)`` -> deep feature vector only (used by the
  fusion phase and by TCAV).
* ``get_attention_maps(x)`` -> a per-voxel saliency map from the
  deepest Swin stage (saved to ``attention/`` by
  :func:`src.utils.visualization.plot_attention_heatmap`).

Failure modes
-------------
* Missing MONAI -> ``ImportError`` on first instantiation.
* Mismatched ``img_size`` vs. dataset cropping -> the dataset resizes
  via trilinear interpolation (see ``HCCDataset._load_volume``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SwinViT3D(nn.Module):
    """Wraps MONAI's 3D Swin Transformer for HCC risk classification.

    The model returns a single deep feature vector per volume which is
    consumed both by an internal classification head (Phase 4 training)
    and by the cross-attention fusion module (Phase 5).
    """

    def __init__(
        self,
        img_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
        depths: tuple[int, ...],
        num_heads: tuple[int, ...],
        window_size: tuple[int, int, int],
        mlp_ratio: float,
        drop_path_rate: float,
        dropout: float,
        num_classes: int,
        use_checkpoint: bool,
    ) -> None:
        """Initialise a 3D Swin Transformer for HCC risk classification.

        All hyperparameters come from the ``swin_vit`` block of the
        config; nothing is hardcoded so ablations can sweep cleanly.

        Args:
            img_size: Spatial size ``(D, H, W)`` of the input volume.
            patch_size: Patch size in voxels.
            in_channels: Number of input channels (1 for grayscale CT).
            embed_dim: Base embedding dimension; later stages double it.
            depths: Number of Swin blocks per stage.
            num_heads: Number of attention heads per stage.
            window_size: Local-attention window size.
            mlp_ratio: Hidden / embed ratio inside each Swin MLP.
            drop_path_rate: Stochastic-depth rate.
            dropout: Drop / attn-drop rate shared across stages.
            num_classes: Output classes for the internal head.
        """
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer

        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.embed_dim = int(embed_dim)
        self.depths = tuple(depths)
        self.num_heads = tuple(num_heads)
        self.window_size = tuple(window_size)
        self.mlp_ratio = float(mlp_ratio)
        self.drop_path_rate = float(drop_path_rate)
        self.num_classes = int(num_classes)
        self.use_checkpoint = bool(use_checkpoint)
        if any(i % p != 0 for i, p in zip(self.img_size, self.patch_size)):
            raise ValueError(
                f"img_size must be divisible by patch_size. Got img_size={self.img_size}, "
                f"patch_size={self.patch_size}."
            )
        tokens_per_dim = tuple(i // p for i, p in zip(self.img_size, self.patch_size))
        if any(t % w != 0 for t, w in zip(tokens_per_dim, self.window_size)):
            raise ValueError(
                "Invalid Swin window setup: (img_size/patch_size) must be divisible by "
                f"window_size. Got tokens_per_dim={tokens_per_dim}, "
                f"window_size={self.window_size}."
            )

        self.encoder = SwinTransformer(
            in_chans=in_channels,
            embed_dim=self.embed_dim,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=list(self.depths),
            num_heads=list(self.num_heads),
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            drop_path_rate=self.drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=3,
        )

        self.feature_dim = self.embed_dim * (2 ** (len(self.depths) - 1))
        self.norm = nn.LayerNorm(self.feature_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )
        self._last_attention: torch.Tensor | None = None

    def _pool_features(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """Take the deepest feature map and global-average-pool it."""
        deepest = hidden_states[-1]
        # MONAI returns NCDHW; pool over spatial dims.
        pooled = deepest.mean(dim=(2, 3, 4))
        return self.norm(pooled)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only the pooled deep feature vector."""
        hidden = self.encoder(x)
        return self._pool_features(hidden)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, deep_feature_vector)``."""
        hidden = self.encoder(x)
        deep = self._pool_features(hidden)
        logits = self.head(deep)
        # cache the deepest hidden state for attention map extraction
        self._last_attention = hidden[-1].detach()
        return logits, deep

    def get_attention_maps(
        self,
        x: torch.Tensor,
        upsample: bool = False,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Return a 3D self-attention saliency map from the deepest Swin stage.

        This is **not** Grad-CAM. It uses the magnitudes of the deepest
        Swin Transformer stage's hidden activations as a proxy for
        per-voxel attention. The motivation: each Swin stage is built
        from windowed self-attention layers, so the activation magnitude
        of the last stage already integrates the model's attention
        decisions over all earlier layers.

        Pipeline stage: K (Self-attention map) in the A->M flow.

        Parameters
        ----------
        x:
            Input mini-batch of shape ``(B, C, D, H, W)``.
        upsample:
            If ``True``, trilinear-upsample attention to ``self.img_size``.
        normalize:
            If ``True``, apply per-sample min-max scaling to ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(B, D', H', W')`` (the deepest stage's
            spatial resolution) where each voxel holds the
            channel-mean of absolute activations.

        Notes
        -----
        - Use :func:`src.utils.visualization.plot_attention_heatmap`
          to upsample this volume back to the original CT space using
          ``crop_metadata.json``.
        - Saved heatmaps live in ``attention/<PID>_attention.nii.gz``.
        """
        with torch.no_grad():
            hidden = self.encoder(x)
        deep = hidden[-1]
        attention = deep.abs().mean(dim=1, keepdim=True)
        if normalize:
            att_min = attention.amin(dim=(2, 3, 4), keepdim=True)
            att_max = attention.amax(dim=(2, 3, 4), keepdim=True)
            attention = (attention - att_min) / (att_max - att_min + 1e-8)
        if upsample:
            attention = F.interpolate(
                attention,
                size=self.img_size,
                mode="trilinear",
                align_corners=False,
            )
        return attention.squeeze(1)
