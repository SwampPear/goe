from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

__all__ = ["InputStem3D", "ConvNormAct3D", "RobustPerChannelNorm3D"]

class RobustPerChannelNorm3D(nn.Module):
    """
    Robust per-batch, per-channel normalization for 3D tensors.

    Steps (per sample, per channel):
      1) Clip to robust percentiles (e.g., 0.5–99.5) over spatial dims (D,H,W).
      2) Standardize to zero-mean, unit-variance.

    Args:
      q_low:  lower quantile in [0,1] (e.g., 0.005 for 0.5%)
      q_high: upper quantile in [0,1] (e.g., 0.995 for 99.5%)
      eps:    numerical stability for std division

    Notes:
      - This is compute-heavy due to torch.quantile; use only if needed at runtime.
      - If you already normalize in preprocessing, set `enable=False` or skip this module.
    """
    def __init__(self, q_low: float = 0.005, q_high: float = 0.995, eps: float = 1e-6, enable: bool = True):
        super().__init__()
        assert 0.0 <= q_low < q_high <= 1.0
        self.q_low = q_low
        self.q_high = q_high
        self.eps = eps
        self.enable = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return x
        # x: [B,C,D,H,W]
        assert x.dim() == 5, f"Expected 5D tensor [B,C,D,H,W], got {x.shape}"
        spatial_dims = (2, 3, 4)
        q_lo = torch.quantile(x, self.q_low, dim=spatial_dims, keepdim=True)
        q_hi = torch.quantile(x, self.q_high, dim=spatial_dims, keepdim=True)
        x = x.clamp(q_lo, q_hi)
        mean = x.mean(dim=spatial_dims, keepdim=True)
        std = x.std(dim=spatial_dims, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std


class ConvNormAct3D(nn.Module):
    """
    A 3D Conv -> GroupNorm -> Activation block that preserves spatial size with padding=1.

    Args:
      in_ch:   input channels
      out_ch:  output channels
      k:       kernel size (default 3)
      s:       stride (default 1)
      g:       groups for GroupNorm (default 8)
      act:     activation: 'silu' (default), 'relu', or 'gelu'
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 8, act: str = "silu"):
        super().__init__()
        pad = (k - 1) // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, bias=False)
        # Choose number of groups that divides out_ch; fallback to 1 if needed
        g_eff = g if out_ch % g == 0 else 1
        self.norm = nn.GroupNorm(g_eff, out_ch)
        if act == "silu":
            self.act = nn.SiLU(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act}")

        # Kaiming init for conv
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class InputStem3D(nn.Module):
    """
    5.1 Input Stage — 3D convolutional stem.

    Expects an input voxel patch tensor x ∈ R[B, C, D, H, W]
    (C ≈ 12-18 engineered feature channels). Applies two Conv3D→GN→SiLU
    blocks to mix channels into a 64-d latent representation while
    preserving spatial resolution.

    Args:
      in_ch:            number of input channels (features)
      mid_ch:           latent channels (default 64)
      repeats:          number of ConvNormAct blocks (default 2; typically 2)
      groups:           GroupNorm groups (default 8)
      activation:       'silu' | 'relu' | 'gelu' (default 'silu')
      robust_norm:      enable RobustPerChannelNorm3D inside stem (default False)
      q_low/high/eps:   percentile/epsilon params for robust normalization

    Notes:
      - Use mixed precision (bfloat16) from your training loop, e.g.:
          with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
              y = stem(x)
      - Input/Output shapes:
          in : [B, C,   D,   H,   W]
          out: [B, 64,  D,   H,   W]  (if mid_ch=64 and repeats≥1)
    """
    def __init__(
        self,
        in_ch: int,
        mid_ch: int = 64,
        repeats: int = 2,
        groups: int = 8,
        activation: str = "silu",
        robust_norm: bool = False,
        q_low: float = 0.005,
        q_high: float = 0.995,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert repeats >= 1, "repeats must be ≥ 1"

        self.robust = RobustPerChannelNorm3D(q_low, q_high, eps, enable=robust_norm)

        blocks = []
        # First block: C → mid_ch
        blocks.append(ConvNormAct3D(in_ch, mid_ch, k=3, s=1, g=groups, act=activation))
        # Subsequent (repeats-1) blocks: mid_ch → mid_ch
        for _ in range(repeats - 1):
            blocks.append(ConvNormAct3D(mid_ch, mid_ch, k=3, s=1, g=groups, act=activation))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optional robust clipping + standardization (if not already done offline)
        x = self.robust(x)
        # Two (or more) Conv3D→GN→Act blocks
        return self.blocks(x)


# ---- Quick self-test (optional) ----
if __name__ == "__main__":
    B, C, D, H, W = 2, 16, 32, 160, 160
    x = torch.randn(B, C, D, H, W)
    stem = InputStem3D(in_ch=C, mid_ch=64, repeats=2, robust_norm=False)
    y = stem(x)
    print("Input :", x.shape)
    print("Output:", y.shape)  # expect [2, 64, 32, 160, 160]
