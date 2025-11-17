from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


__all__ = [
    "binary_dice",
    "binary_iou",
    "abs_rel",
    "rmse",
    "MetricMeter",
    "MetricCollection",
]


# ---------------------------------------------------------------------------
# Basic segmentation / regression metrics
# ---------------------------------------------------------------------------


def _to_bool_mask(mask: Optional[Tensor], ref: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    if mask.shape != ref.shape:
        try:
            mask = mask.expand_as(ref)
        except Exception as e:  # pragma: no cover
            raise ValueError(
                f"mask shape {tuple(mask.shape)} not broadcastable to {tuple(ref.shape)}"
            ) from e
    return mask.bool()


def binary_dice(
    logits_or_probs: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Compute Dice coefficient for binary segmentation.

    Parameters
    ----------
    logits_or_probs : Tensor
        Either logits or probabilities, arbitrary shape (B, ...).
    target : Tensor
        Binary target in {0,1} or [0,1], same shape.
    threshold : float
        Threshold applied AFTER sigmoid if logits are supplied.
        If your input is already probabilities, pass values ∈ [0,1] and this
        will threshold them directly.
    eps : float
        Numerical stability.
    mask : Tensor, optional
        Binary mask of same shape or broadcastable; 0 → ignore voxel.

    Returns
    -------
    float
        Dice coefficient in [0,1].
    """
    x = logits_or_probs

    # Heuristic: treat as logits if values span > 1 or outside [0,1].
    if x.min() < 0.0 or x.max() > 1.0:
        x = torch.sigmoid(x)

    x = (x >= threshold).to(torch.float32)
    y = (target > 0.5).to(torch.float32)

    m = _to_bool_mask(mask, y)
    if m is not None:
        x = x[m]
        y = y[m]

    intersection = (x * y).sum()
    union = x.sum() + y.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.item())


def binary_iou(
    logits_or_probs: Tensor,
    target: Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Intersection-over-Union (IoU) for binary segmentation.

    Same conventions as `binary_dice`.
    """
    x = logits_or_probs

    if x.min() < 0.0 or x.max() > 1.0:
        x = torch.sigmoid(x)

    x = (x >= threshold).to(torch.float32)
    y = (target > 0.5).to(torch.float32)

    m = _to_bool_mask(mask, y)
    if m is not None:
        x = x[m]
        y = y[m]

    intersection = (x * y).sum()
    union = x.sum() + y.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return float(iou.item())


def abs_rel(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> float:
    """
    Absolute relative error for geometry / depth regression.

        AbsRel = |pred - target| / (target + eps)

    Parameters
    ----------
    pred : Tensor
        Predictions, arbitrary shape.
    target : Tensor
        Ground-truth, same shape.
    mask : Tensor, optional
        Binary mask, same shape or broadcastable.
    eps : float
        Stability constant added to denominator.

    Returns
    -------
    float
        Mean absolute relative error.
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)

    if mask is not None:
        m = _to_bool_mask(mask, target)
        pred = pred[m]
        target = target[m]

    denom = target.abs() + eps
    rel = (pred - target).abs() / denom
    return float(rel.mean().item())


def rmse(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Root-mean-square error for regression targets.
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)

    if mask is not None:
        m = _to_bool_mask(mask, target)
        pred = pred[m]
        target = target[m]

    mse = torch.mean((pred - target) ** 2)
    return float(torch.sqrt(mse).item())


# ---------------------------------------------------------------------------
# Running meters
# ---------------------------------------------------------------------------


@dataclass
class MetricMeter:
    """
    Simple running average for a scalar metric.

    Example
    -------
    >>> m = MetricMeter()
    >>> m.update(0.5, n=10)
    >>> m.update(0.7, n=5)
    >>> m.avg
    """
    total: float = 0.0
    count: float = 0.0

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * float(n)
        self.count += float(n)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0.0


class MetricCollection:
    """
    A small helper to track multiple named metrics.

    Example
    -------
    >>> metrics = MetricCollection(["loss", "dice"])
    >>> metrics.update("loss", 0.3, n=16)
    >>> metrics.update("dice", 0.8, n=16)
    >>> metrics.as_dict()
    {'loss': 0.3, 'dice': 0.8}
    """

    def __init__(self, names):
        self._meters: Dict[str, MetricMeter] = {n: MetricMeter() for n in names}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self._meters:
            self._meters[name] = MetricMeter()
        self._meters[name].update(value, n)

    def reset(self) -> None:
        for m in self._meters.values():
            m.reset()

    def as_dict(self) -> Dict[str, float]:
        return {k: v.avg for k, v in self._meters.items()}
