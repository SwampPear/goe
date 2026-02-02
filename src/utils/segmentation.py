from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

__all__ = [
    "binarize_logits",
    "resize_logits",
]


def binarize_logits(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """
    Convert logits/probabilities to a binary mask (0/1 float tensor).
    """
    x = logits
    if x.min() < 0.0 or x.max() > 1.0:
        x = torch.sigmoid(x)
    return (x >= threshold).to(torch.float32)


def resize_logits(
    logits: Tensor,
    out_shape: Tuple[int, int, int],
    mode: str = "trilinear",
) -> Tensor:
    """
    Resize logits [B, C, D, H, W] to a target spatial shape.
    """
    if logits.dim() != 5:
        raise ValueError(f"logits must be 5D [B, C, D, H, W], got {tuple(logits.shape)}")
    return F.interpolate(
        logits,
        size=out_shape,
        mode=mode,
        align_corners=False if mode in ("trilinear", "bilinear") else None,
    )
