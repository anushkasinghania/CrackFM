"""Segmentation losses tuned for thin, class-imbalanced crack masks.

Combines Dice (region overlap), BCE (per-pixel), Tversky (recall-biased for the
rare foreground) and a boundary term (penalises errors near GT edges, which
matters when the structure is only a few pixels wide).

Requires PyTorch; import lazily so the rest of the package works without it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (num + EPS) / (den + EPS)).mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, beta: float = 0.7) -> torch.Tensor:
    """Tversky loss; beta>0.5 weights false negatives more (recall-biased)."""
    prob = torch.sigmoid(logits)
    tp = (prob * target).sum(dim=(1, 2, 3))
    fp = (prob * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - prob) * target).sum(dim=(1, 2, 3))
    tversky = (tp + EPS) / (tp + (1 - beta) * fp + beta * fn + EPS)
    return (1 - tversky).mean()


def _boundary_weight(target: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Higher weight on pixels near the GT boundary (dilated XOR eroded)."""
    k = 2 * width + 1
    pad = width
    dil = F.max_pool2d(target, kernel_size=k, stride=1, padding=pad)
    ero = -F.max_pool2d(-target, kernel_size=k, stride=1, padding=pad)
    band = (dil - ero).clamp(0, 1)
    return 1.0 + band  # 1 elsewhere, 2 in the boundary band


def boundary_loss(logits: torch.Tensor, target: torch.Tensor, width: int = 3) -> torch.Tensor:
    w = _boundary_weight(target, width)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * w).mean()


class CrackLoss(torch.nn.Module):
    """Weighted sum of Dice + BCE + boundary, with Tversky folded into Dice."""

    def __init__(
        self,
        dice: float = 1.0,
        bce: float = 0.5,
        boundary: float = 0.25,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        self.w_dice = dice
        self.w_bce = bce
        self.w_boundary = boundary
        self.tversky_beta = tversky_beta

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        loss = logits.new_zeros(())
        if self.w_dice:
            loss = loss + self.w_dice * tversky_loss(logits, target, self.tversky_beta)
        if self.w_bce:
            loss = loss + self.w_bce * F.binary_cross_entropy_with_logits(logits, target)
        if self.w_boundary:
            loss = loss + self.w_boundary * boundary_loss(logits, target)
        return loss
