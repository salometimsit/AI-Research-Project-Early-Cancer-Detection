"""Classification head and Focal Loss implementation.

Pipeline stage: M in the A->M flow.

Inputs
------
* Fused feature vector from
  :class:`src.models.fusion.CrossAttentionFusion`.
* ``training.focal_loss.alpha`` and ``training.focal_loss.gamma``
  from the config.

Outputs
-------
* ``HCCClassifier`` -> 2-class logits.
* ``FusedHCCModel`` -> end-to-end wrapper combining SwinViT + fusion +
  head, used during inference.
* ``FocalLoss`` -> scalar loss value handling class imbalance.

Failure modes
-------------
* Fused dim mismatch -> shape mismatch raised by the linear layer.
* Logits/labels size mismatch in ``FocalLoss.forward`` -> standard
  PyTorch broadcast errors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as fnn


class HCCClassifier(nn.Module):
    """Final classification head producing 2-class logits."""

    def __init__(self, fused_dim: int, dropout: float, num_classes: int) -> None:
        """Build a 2-layer GELU MLP head.

        Args:
            fused_dim: Dimensionality of the fused feature vector.
            dropout: Dropout probability applied between layers.
            num_classes: Number of output logits (2 for HCC vs control).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """Map a ``(B, fused_dim)`` tensor to ``(B, num_classes)`` logits."""
        return self.net(fused_features)


class FusedHCCModel(nn.Module):
    """End-to-end fused model: SwinViT + radiomics -> cross-attention -> head.

    Used at inference time. Training composes these modules manually so
    radiomics features can be precomputed once.
    """

    def __init__(self, swin: nn.Module, fusion: nn.Module, classifier: nn.Module) -> None:
        """Wire three already-constructed sub-modules together.

        Args:
            swin: A SwinViT (or any module) whose ``forward(volume)``
                returns ``(logits, deep_features)``.
            fusion: Cross-attention fusion module taking
                ``(radiomics_vec, deep_vec)``.
            classifier: Head mapping fused features to logits.
        """
        super().__init__()
        self.swin = swin
        self.fusion = fusion
        self.classifier = classifier

    def forward(
        self,
        volume: torch.Tensor,
        radiomics_features: torch.Tensor,
    ) -> torch.Tensor:
        _, deep = self.swin(volume)
        fused = self.fusion(radiomics_features, deep)
        return self.classifier(fused)


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    Accepts logits of shape ``(N, 2)`` and integer targets of shape ``(N,)``.
    """

    def __init__(self, alpha: float, gamma: float, reduction: str) -> None:
        """Initialise the focal loss.

        Args:
            alpha: Class-1 weight in ``[0, 1]``. Class-0 receives
                ``1 - alpha``. Recommended: compute from training-set
                imbalance (``alpha = N_neg / (N_neg + N_pos)``).
            gamma: Focusing parameter; higher = more down-weighting of
                easy examples.
            reduction: One of ``"mean"``, ``"sum"``, or ``"none"``.
        """
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss given ``(N, 2)`` logits and ``(N,)`` targets."""
        ce = fnn.cross_entropy(logits, targets, reduction="none")
        log_p = fnn.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)
        alpha_pos = torch.tensor(self.alpha, dtype=pt.dtype, device=pt.device)
        alpha_neg = torch.tensor(1.0 - self.alpha, dtype=pt.dtype, device=pt.device)
        alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
        loss = alpha_t * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
