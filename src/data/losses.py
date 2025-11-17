from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


__all__ = [
    # functional losses
    "bce_with_logits_masked",
    "dice_loss_from_logits",
    "ink_loss",
    "geometry_l1_loss",
    "geometry_smooth_l1_loss",
    # heads
    "InkHead",
    "GeometryHead",
    "MultiTaskHead",
    # configs / wrappers
    "InkLossConfig",
    "GeometryLossConfig",
    "MultiTaskLoss",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast_mask(mask: Tensor, ref: Tensor) -> Tensor:
    """
    Make `mask` broadcastable to `ref` by unsqueezing trailing dims.

    mask : (B, ...)  or (B, N, ...)
    ref  : (B, C, ...) or (B, N, C, ...) or similar

    Returns a float mask with same ndim as ref.
    """
    if mask.dim() > ref.dim():
        raise ValueError(
            f"mask.ndim ({mask.dim()}) cannot exceed ref.ndim ({ref.dim()})"
        )

    out = mask
    while out.dim() < ref.dim():
        out = out.unsqueeze(-1)
    return out.to(dtype=ref.dtype)


# ---------------------------------------------------------------------------
# Ink (segmentation) losses
# ---------------------------------------------------------------------------


def bce_with_logits_masked(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Binary cross-entropy with logits, with optional spatial/voxel mask.

    Parameters
    ----------
    logits : Tensor
        Raw logits, arbitrary shape, e.g. (B, 1, D, H, W) or (B, N, 1).
    targets : Tensor
        Binary targets in {0, 1}, same shape as logits or broadcastable.
    mask : Tensor, optional
        Binary mask, shape broadcastable to `targets`. Positions with mask==0
        are ignored.
    pos_weight : Tensor, optional
        Class balancing weight as in torch.nn.BCEWithLogitsLoss.
    reduction : {"none", "mean", "sum"}
        Reduction over all elements.

    Returns
    -------
    Tensor
        Scalar loss if reduction != "none", else per-element tensor.
    """
    if mask is None:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=reduction
        )

    m = _broadcast_mask(mask, targets)
    # Avoid computing BCE on masked-out locations by zeroing both logits and
    # targets and later rescaling.
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    loss = loss * m

    if reduction == "none":
        return loss

    denom = m.sum().clamp_min(1.0)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.sum() / denom
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def dice_loss_from_logits(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Soft Dice loss for binary segmentation, computed from logits.

    Handles arbitrary shapes, flattening over all dims except batch.
    `targets` should be {0,1} or in [0,1]. Mask (if given) masks out voxels.

    Parameters
    ----------
    logits : Tensor
        Raw logits, shape (B, ...) such as (B, 1, D, H, W) or (B, N, 1).
    targets : Tensor
        Binary targets, same shape as logits or broadcastable.
    mask : Tensor, optional
        Binary mask, shape broadcastable to targets. 0 -> ignore.
    eps : float
        Numerical stability constant.

    Returns
    -------
    Tensor
        Scalar Dice loss (1 - Dice).
    """
    probs = torch.sigmoid(logits)
    probs = probs.float()
    targets = targets.float()
    if mask is not None:
        m = _broadcast_mask(mask, targets)
        probs = probs * m
        targets = targets * m

    B = logits.shape[0]
    probs_flat = probs.view(B, -1)
    targets_flat = targets.view(B, -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice  # (B,)
    return loss.mean()


def ink_loss(
    logits: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: Optional[Tensor] = None,
) -> Tensor:
    """
    Combined ink segmentation loss: λ_b * BCE + λ_d * Dice.

    This is the main loss for ink detection (binary segmentation).

    Parameters
    ----------
    logits : Tensor
        Ink logits, shape (B, ...) e.g. (B, 1, D, H, W) or (B, N, 1).
    targets : Tensor
        Binary ink labels, same shape as logits or broadcastable.
    mask : Tensor, optional
        Binary mask over voxels/tokens, broadcastable to targets.
    bce_weight : float
        Weight for BCE component λ_b.
    dice_weight : float
        Weight for Dice component λ_d.
    pos_weight : Tensor, optional
        Positive class weight for BCE, as in BCEWithLogitsLoss.

    Returns
    -------
    Tensor
        Scalar combined loss.
    """
    if bce_weight == 0.0 and dice_weight == 0.0:
        raise ValueError("At least one of bce_weight or dice_weight must be non-zero.")

    loss_bce = bce_with_logits_masked(
        logits, targets, mask=mask, pos_weight=pos_weight, reduction="mean"
    ) if bce_weight != 0.0 else logits.new_tensor(0.0)

    loss_dice = dice_loss_from_logits(
        logits, targets, mask=mask
    ) if dice_weight != 0.0 else logits.new_tensor(0.0)

    return bce_weight * loss_bce + dice_weight * loss_dice


# ---------------------------------------------------------------------------
# Geometry losses (e.g., depth / surface height / normals)
# ---------------------------------------------------------------------------


def geometry_l1_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Masked L1 loss for geometry regression targets.

    Useful for supervising depth / surface height / coordinate predictions.

    Parameters
    ----------
    pred : Tensor
        Predicted geometry, arbitrary shape (B, ...).
    target : Tensor
        Ground-truth geometry, same shape as pred or broadcastable.
    mask : Tensor, optional
        Binary mask, broadcastable to target. 0 -> ignore.
    reduction : {"none", "mean", "sum"}
        Reduction over elements.

    Returns
    -------
    Tensor
        Scalar loss if reduction != "none".
    """
    if mask is None:
        return F.l1_loss(pred, target, reduction=reduction)

    m = _broadcast_mask(mask, target)
    loss = F.l1_loss(pred, target, reduction="none") * m

    if reduction == "none":
        return loss

    denom = m.sum().clamp_min(1.0)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.sum() / denom
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def geometry_smooth_l1_loss(
    pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    beta: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Masked Smooth L1 / Huber loss for geometry regression.

    Parameters
    ----------
    pred : Tensor
        Predicted geometry, arbitrary shape (B, ...).
    target : Tensor
        Ground-truth geometry, same shape as pred or broadcastable.
    mask : Tensor, optional
        Binary mask, broadcastable to target. 0 -> ignore.
    beta : float
        Transition point between L2 and L1, as in SmoothL1Loss.
    reduction : {"none", "mean", "sum"}
        Reduction over elements.

    Returns
    -------
    Tensor
        Scalar loss if reduction != "none".
    """
    if mask is None:
        return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)

    m = _broadcast_mask(mask, target)
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none") * m

    if reduction == "none":
        return loss

    denom = m.sum().clamp_min(1.0)
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.sum() / denom
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


# ---------------------------------------------------------------------------
# Small task heads (token / patch → ink / geometry predictions)
# ---------------------------------------------------------------------------


class InkHead(nn.Module):
    """
    Simple MLP head that maps token features → ink logits.

    Can be used on:
      - per-patch pooled features: (B, N, D)
      - per-voxel embedded features: (B, C, D, H, W) (after reshape).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 0,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        if hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_channels))
        else:
            layers.append(nn.Linear(in_dim, out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input features, shape (B, N, D).

        Returns
        -------
        Tensor
            Ink logits, shape (B, N, out_channels).
        """
        if x.dim() != 3:
            raise ValueError(f"InkHead expects (B, N, D), got {tuple(x.shape)}")
        return self.mlp(x)


class GeometryHead(nn.Module):
    """
    Small MLP head for geometry regression (e.g., depth / surface).

    Same calling convention as InkHead but returns real-valued outputs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 0,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        if hidden_dim > 0:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input features, shape (B, N, D).

        Returns
        -------
        Tensor
            Geometry predictions, shape (B, N, out_dim).
        """
        if x.dim() != 3:
            raise ValueError(f"GeometryHead expects (B, N, D), got {tuple(x.shape)}")
        return self.mlp(x)


class MultiTaskHead(nn.Module):
    """
    Convenience head: shared input → ink logits + geometry predictions.

    Intended for a GoE block that outputs per-token embeddings H ∈ ℝ^{B×N×D}.
    """

    def __init__(
        self,
        in_dim: int,
        ink_hidden_dim: int = 0,
        geom_hidden_dim: int = 0,
        ink_out_channels: int = 1,
        geom_out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.ink_head = InkHead(
            in_dim=in_dim,
            hidden_dim=ink_hidden_dim,
            out_channels=ink_out_channels,
        )
        self.geometry_head = GeometryHead(
            in_dim=in_dim,
            hidden_dim=geom_hidden_dim,
            out_dim=geom_out_dim,
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input features, shape (B, N, D).

        Returns
        -------
        Dict[str, Tensor]
            {
                "ink_logits": (B, N, ink_out_channels),
                "geometry":   (B, N, geom_out_dim),
            }
        """
        ink_logits = self.ink_head(x)
        geom = self.geometry_head(x)
        return {"ink_logits": ink_logits, "geometry": geom}


# ---------------------------------------------------------------------------
# Configs + multi-task wrapper
# ---------------------------------------------------------------------------


@dataclass
class InkLossConfig:
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    pos_weight: Optional[float] = None  # scalar, converted to tensor at runtime


@dataclass
class GeometryLossConfig:
    weight: float = 1.0
    loss_type: str = "smooth_l1"  # {"l1", "smooth_l1"}
    beta: float = 1.0  # for smooth_l1


class MultiTaskLoss(nn.Module):
    """
    Combine ink and geometry losses into a single scalar for training.

    Expected usage inside a training step:

        outputs = model(batch)  # dict
        loss_dict = criterion(outputs, batch)
        loss = loss_dict["loss"]

    Conventions
    -----------
    `outputs` should contain:
      - "ink_logits": (B, ..., 1 or C_ink)
      - "geometry":   (B, ..., G)

    `batch` should contain:
      - "ink_target": same shape as "ink_logits" (or broadcastable)
      - "ink_mask":   optional mask
      - "geometry_target": same shape as "geometry"
      - "geometry_mask":   optional mask
    """

    def __init__(
        self,
        ink_cfg: InkLossConfig = InkLossConfig(),
        geom_cfg: GeometryLossConfig = GeometryLossConfig(),
    ) -> None:
        super().__init__()
        self.ink_cfg = ink_cfg
        self.geom_cfg = geom_cfg

    def forward(
        self,
        outputs: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute multi-task loss.

        Returns
        -------
        Dict[str, Tensor]
            {
                "loss":         total loss (scalar),
                "loss_ink":     ink loss (scalar, may be zero),
                "loss_geom":    geometry loss (scalar, may be zero),
            }
        """
        total_loss = outputs["ink_logits"].new_tensor(0.0)

        # ----- Ink loss -----
        loss_ink = outputs["ink_logits"].new_tensor(0.0)
        if "ink_logits" in outputs and "ink_target" in batch:
            ink_logits = outputs["ink_logits"]
            ink_target = batch["ink_target"]
            ink_mask = batch.get("ink_mask", None)

            pos_weight_tensor = None
            if self.ink_cfg.pos_weight is not None:
                pos_weight_tensor = torch.tensor(
                    [self.ink_cfg.pos_weight],
                    device=ink_logits.device,
                    dtype=ink_logits.dtype,
                )

            loss_ink = ink_loss(
                ink_logits,
                ink_target,
                mask=ink_mask,
                bce_weight=self.ink_cfg.bce_weight,
                dice_weight=self.ink_cfg.dice_weight,
                pos_weight=pos_weight_tensor,
            )
            total_loss = total_loss + loss_ink

        # ----- Geometry loss -----
        loss_geom = outputs["ink_logits"].new_tensor(0.0)
        if "geometry" in outputs and "geometry_target" in batch:
            geom_pred = outputs["geometry"]
            geom_target = batch["geometry_target"]
            geom_mask = batch.get("geometry_mask", None)

            if self.geom_cfg.loss_type == "l1":
                loss_geom = geometry_l1_loss(
                    geom_pred,
                    geom_target,
                    mask=geom_mask,
                    reduction="mean",
                )
            elif self.geom_cfg.loss_type == "smooth_l1":
                loss_geom = geometry_smooth_l1_loss(
                    geom_pred,
                    geom_target,
                    mask=geom_mask,
                    beta=self.geom_cfg.beta,
                    reduction="mean",
                )
            else:
                raise ValueError(f"Unsupported geometry loss_type: {self.geom_cfg.loss_type}")

            loss_geom = self.geom_cfg.weight * loss_geom
            total_loss = total_loss + loss_geom

        return {
            "loss": total_loss,
            "loss_ink": loss_ink,
            "loss_geom": loss_geom,
        }
