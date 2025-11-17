from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import Tensor, nn


__all__ = [
    "RandomFlip3D",
    "RandomRotate90HW",
    "IntensityJitter3D",
    "GaussianNoise3D",
    "VesuviusAugmentations",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SPATIAL_KEYS_DEFAULT = (
    "volume",
    "ink_target",
    "geometry_target",
    "ink_mask",
    "geometry_mask",
)


def _apply_same_spatial(
    sample: Dict[str, Tensor],
    fn,
    keys: Sequence[str] = _SPATIAL_KEYS_DEFAULT,
) -> Dict[str, Tensor]:
    """
    Apply a spatial operation fn(x) to all tensors that share the same
    3D shape (B, C, D, H, W) or (B, 1, D, H, W).

    fn should take a tensor and return a tensor of the same shape.
    """
    for k in keys:
        if k not in sample:
            continue
        x = sample[k]
        # Only apply to 5D tensors with spatial dims.
        if x.dim() == 5:
            sample[k] = fn(x)
    return sample


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------


class RandomFlip3D(nn.Module):
    """
    Random 3D flips along depth / height / width axes.

    Expects tensors of shape (B, C, D, H, W).

    Parameters
    ----------
    p : float
        Probability of applying *each* axis flip independently.
    axes : Iterable[int]
        Spatial axes to consider, as indices into the 5D tensor.
        Default (2, 3, 4) ≡ (D, H, W).
    keys : sequence of str
        Sample dict keys to which to apply the spatial transform.
    """

    def __init__(
        self,
        p: float = 0.5,
        axes: Iterable[int] = (2, 3, 4),
        keys: Sequence[str] = _SPATIAL_KEYS_DEFAULT,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.axes = tuple(int(a) for a in axes)
        self.keys = tuple(keys)

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        def _flip(x: Tensor) -> Tensor:
            axes_to_flip: List[int] = []
            for a in self.axes:
                if torch.rand(()) < self.p:
                    axes_to_flip.append(a)
            if not axes_to_flip:
                return x
            return torch.flip(x, dims=axes_to_flip)

        return _apply_same_spatial(sample, _flip, keys=self.keys)


class RandomRotate90HW(nn.Module):
    """
    Random 90° rotations in the H×W plane (in-plane rotations).

    Expects tensors of shape (B, C, D, H, W). For each sample, with
    probability p, rotate by k * 90° where k ∈ {1,2,3} uniformly.

    Parameters
    ----------
    p : float
        Probability of applying a rotation at all.
    keys : sequence of str
        Sample dict keys to which to apply the spatial transform.
    """

    def __init__(
        self,
        p: float = 0.5,
        keys: Sequence[str] = _SPATIAL_KEYS_DEFAULT,
    ) -> None:
        super().__init__()
        self.p = float(p)
        self.keys = tuple(keys)

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if torch.rand(()) >= self.p:
            return sample

        k = int(torch.randint(1, 4, (1,)).item())  # 1, 2, or 3

        def _rot(x: Tensor) -> Tensor:
            # x: (B, C, D, H, W) -> rotate H/W dims
            B, C, D, H, W = x.shape
            # Flatten B*C*D as one batch dimension, rotate 2D, then reshape back.
            x_flat = x.view(B * C * D, H, W)
            x_rot = torch.rot90(x_flat, k=k, dims=(-2, -1))
            H2, W2 = x_rot.shape[-2:]
            return x_rot.view(B, C, D, H2, W2)

        return _apply_same_spatial(sample, _rot, keys=self.keys)


class IntensityJitter3D(nn.Module):
    """
    Random intensity jitter for volumes: multiplicative gain + additive bias.

    Applied only to `volume` by default.

    volume := volume * g + b

    Parameters
    ----------
    gain_range : (float, float)
        Range for multiplicative gain g sampled uniformly.
    bias_range : (float, float)
        Range for additive bias b sampled uniformly.
    p : float
        Probability of applying jitter.
    key : str
        Key in sample dict to apply jitter to (default: "volume").
    """

    def __init__(
        self,
        gain_range: Sequence[float] = (0.9, 1.1),
        bias_range: Sequence[float] = (-0.05, 0.05),
        p: float = 0.8,
        key: str = "volume",
    ) -> None:
        super().__init__()
        if len(gain_range) != 2 or len(bias_range) != 2:
            raise ValueError("gain_range and bias_range must be length-2 sequences.")
        self.gain_range = (float(gain_range[0]), float(gain_range[1]))
        self.bias_range = (float(bias_range[0]), float(bias_range[1]))
        self.p = float(p)
        self.key = key

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.key not in sample:
            return sample
        if torch.rand(()) >= self.p:
            return sample

        x = sample[self.key]
        if x.dim() != 5:
            return sample  # only apply to volumes

        g = torch.empty((), device=x.device).uniform_(*self.gain_range)
        b = torch.empty((), device=x.device).uniform_(*self.bias_range)

        # Broadcast g, b over all dims.
        sample[self.key] = x * g + b
        return sample


class GaussianNoise3D(nn.Module):
    """
    Additive Gaussian noise to the volume.

    volume := volume + σ * N(0, 1)

    Parameters
    ----------
    sigma : float
        Standard deviation of noise.
    p : float
        Probability of applying noise.
    key : str
        Key in sample dict to apply noise to (default: "volume").
    """

    def __init__(
        self,
        sigma: float = 0.01,
        p: float = 0.8,
        key: str = "volume",
    ) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.p = float(p)
        self.key = key

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.key not in sample:
            return sample
        if torch.rand(()) >= self.p:
            return sample

        x = sample[self.key]
        if x.dim() != 5:
            return sample

        noise = torch.randn_like(x) * self.sigma
        sample[self.key] = x + noise
        return sample


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


@dataclass
class VesuviusAugConfig:
    """
    Simple configuration for default Vesuvius 3D augmentations.

    This is intentionally lightweight; you can extend it or bypass it
    entirely and build your own nn.Sequential of augmentations.
    """

    flip_p: float = 0.5
    rotate_p: float = 0.5
    jitter_p: float = 0.8
    jitter_gain_range: Sequence[float] = (0.9, 1.1)
    jitter_bias_range: Sequence[float] = (-0.05, 0.05)
    noise_p: float = 0.8
    noise_sigma: float = 0.01


class VesuviusAugmentations(nn.Module):
    """
    Default composable augmentation stack for Vesuvius scroll patches.

    Expects and returns a sample dict of the form:

        sample = {
            "volume":          (B, C, D, H, W),      # required
            "ink_target":      (B, 1, D, H, W),      # optional
            "geometry_target": (B, G, D, H, W),      # optional
            "ink_mask":        (B, 1, D, H, W),      # optional
            "geometry_mask":   (B, 1, D, H, W),      # optional
            ...
        }

    All spatial transforms (flip, rotate) are applied consistently across
    all of these keys; intensity and noise by default only affect "volume".
    """

    def __init__(self, cfg: VesuviusAugConfig = VesuviusAugConfig()) -> None:
        super().__init__()
        self.cfg = cfg

        self.transforms = nn.ModuleList(
            [
                RandomFlip3D(p=cfg.flip_p),
                RandomRotate90HW(p=cfg.rotate_p),
                IntensityJitter3D(
                    gain_range=cfg.jitter_gain_range,
                    bias_range=cfg.jitter_bias_range,
                    p=cfg.jitter_p,
                    key="volume",
                ),
                GaussianNoise3D(
                    sigma=cfg.noise_sigma,
                    p=cfg.noise_p,
                    key="volume",
                ),
            ]
        )

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for t in self.transforms:
            sample = t(sample)
        return sample
