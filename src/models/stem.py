# stem.py
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch import Tensor


@dataclass
class InputStemConfig:
    """
    Configuration for the InputStem.

    Attributes
    ----------
    in_channels:
        Number of input channels in the tomographic volume (C_in).
    stem_channels:
        Number of feature channels produced by the 3D encoder φ(x; θ_s).
    token_dim:
        Dimensionality of token embeddings t_i.
    aux_dim:
        Dimensionality of auxiliary routing features a_i.
    patch_size:
        Size of each 3D patch (kD, kH, kW) in voxels.
    patch_stride:
        Stride between consecutive patches (sD, sH, sW). Can be smaller than
        patch_size to allow overlapping patches.
    use_layer_norm:
        Whether to apply LayerNorm to token embeddings.
    """
    in_channels: int = 1
    stem_channels: int = 32
    token_dim: int = 128
    aux_dim: int = 32
    patch_size: Tuple[int, int, int] = (16, 16, 16)
    patch_stride: Tuple[int, int, int] = (8, 8, 8)
    use_layer_norm: bool = True


class Conv3DEncoder(nn.Module):
    """
    Lightweight 3D convolutional encoder φ(x; θ_s).

    This module extracts localized features such as carbonization gradients,
    fiber structure, and ink-related density anomalies from the raw volume.
    """

    def __init__(self, in_channels: int, stem_channels: int):
        super().__init__()

        # A simple but expressive 3D CNN: two conv blocks with downsampling.
        # You can swap this for a deeper encoder (ResNet, UNet, etc.) if needed.
        mid = stem_channels // 2

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),

            # Optional downsampling to aggregate local context.
            nn.Conv3d(mid, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Input volume of shape [B, C_in, D, H, W].

        Returns
        -------
        f0:
            Feature volume of shape [B, C_stem, D', H', W'].
        """
        return self.encoder(x)


def _compute_output_grid(
    D: int,
    H: int,
    W: int,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    kD, kH, kW = patch_size
    sD, sH, sW = stride

    out_D = math.floor((D - kD) / sD) + 1
    out_H = math.floor((H - kH) / sH) + 1
    out_W = math.floor((W - kW) / sW) + 1
    return out_D, out_H, out_W


def extract_patches_3d(
    features: Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> Tuple[Tensor, Tensor]:
    """
    Extract overlapping 3D patches from a feature volume.

    Implements ψ(f0, Ω_i) by unfolding the [D, H, W] dimensions into a
    collection of local voxels.

    Parameters
    ----------
    features:
        Feature map f0 ∈ R^{B×C×D×H×W}.
    patch_size:
        (kD, kH, kW) patch size in voxels.
    stride:
        (sD, sH, sW) stride in voxels.

    Returns
    -------
    patches:
        Tensor of shape [B, N, C * kD * kH * kW] containing flattened patches.
    coords:
        Tensor of shape [B, N, 3] with (z, y, x) coordinates of patch centers
        in normalized [0, 1] coordinates relative to the feature volume.
    """
    B, C, D, H, W = features.shape
    kD, kH, kW = patch_size
    sD, sH, sW = stride

    # Unfold along each spatial dimension: result shape
    # [B, C, D_out, H_out, W_out, kD, kH, kW]
    unfolded = (
        features
        .unfold(dimension=2, size=kD, step=sD)
        .unfold(dimension=3, size=kH, step=sH)
        .unfold(dimension=4, size=kW, step=sW)
    )

    B, C, D_out, H_out, W_out, _, _, _ = unfolded.shape

    # Flatten patch dimensions
    patches = unfolded.contiguous().view(
        B, C, D_out * H_out * W_out, kD * kH * kW
    )
    patches = patches.permute(0, 2, 1, 3).contiguous()  # [B, N, C, K]
    patches = patches.view(B, D_out * H_out * W_out, C * kD * kH * kW)  # [B, N, C*K]

    # Compute normalized patch center coordinates for positional information.
    # Centers are in the coordinate system of the feature volume (D, H, W).
    device = features.device
    z_idxs = torch.arange(D_out, device=device) * sD + (kD - 1) / 2.0
    y_idxs = torch.arange(H_out, device=device) * sH + (kH - 1) / 2.0
    x_idxs = torch.arange(W_out, device=device) * sW + (kW - 1) / 2.0

    zz, yy, xx = torch.meshgrid(z_idxs, y_idxs, x_idxs, indexing="ij")  # [D_out, H_out, W_out]

    zz = zz.reshape(-1) / max(D - 1, 1)
    yy = yy.reshape(-1) / max(H - 1, 1)
    xx = xx.reshape(-1) / max(W - 1, 1)

    coords = torch.stack([zz, yy, xx], dim=-1)  # [N, 3]
    coords = coords.unsqueeze(0).repeat(B, 1, 1)  # [B, N, 3]

    return patches, coords


class InputStem(nn.Module):
    """
    Input stem mapping raw 3D X-ray volumes into token embeddings and
    auxiliary routing features as described in Section 2.2 of the paper.

    Given x ∈ R^{B×C×D×H×W}, we compute:
        f0 = φ(x; θ_s)
        ψ(f0, Ω_i) → local patch features
        t_i = W_t ψ(f0, Ω_i) + b_t                (Equation 1)
        a_i = g_aux(f0[Ω_i])                      (Equation 2)

    The output tokens T and aux features a are then consumed by the encoder
    and graph router for compositional reasoning.
    """

    def __init__(self, cfg: InputStemConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Conv3DEncoder(
            in_channels=cfg.in_channels,
            stem_channels=cfg.stem_channels,
        )

        # Projection from flattened patch features to token embeddings.
        self.token_proj: Optional[nn.Linear] = None
        # Projection to auxiliary routing features.
        self.aux_proj: Optional[nn.Linear] = None

        # LayerNorm over token_dim for stability.
        self.token_norm = nn.LayerNorm(cfg.token_dim) if cfg.use_layer_norm else nn.Identity()

        # We lazily initialize the linear layers once we know the spatial
        # dimensions after the encoder, to compute the flattened patch size.
        self._initialized = False

    def _lazy_init_linears(self, f0: Tensor) -> None:
        """
        Initialize token and aux projections after seeing the first feature
        map, so we know the patch volume size C * kD * kH * kW.
        """
        if self._initialized:
            return

        _, C, D, H, W = f0.shape
        kD, kH, kW = self.cfg.patch_size
        sD, sH, sW = self.cfg.patch_stride

        out_D, out_H, out_W = _compute_output_grid(D, H, W, self.cfg.patch_size, self.cfg.patch_stride)
        if out_D <= 0 or out_H <= 0 or out_W <= 0:
            raise ValueError(
                f"Patch configuration yields no patches: "
                f"feat_volume=({D}, {H}, {W}), patch_size={self.cfg.patch_size}, "
                f"stride={self.cfg.patch_stride}"
            )

        flattened_patch_dim = C * kD * kH * kW

        self.token_proj = nn.Linear(flattened_patch_dim, self.cfg.token_dim)
        self.aux_proj = nn.Linear(flattened_patch_dim, self.cfg.aux_dim)

        self._initialized = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Forward pass through the input stem.

        Parameters
        ----------
        x:
            Raw volumetric scan of shape [B, C_in, D, H, W].

        Returns
        -------
        tokens:
            Token embeddings T ∈ R^{B×N×d_token} produced from local 3D patches.
        aux:
            Auxiliary routing features A ∈ R^{B×N×d_aux} describing local physical
            structure (geometry, curvature, anisotropy) for each patch.
        meta:
            Dictionary with metadata that can be used by downstream modules.
            Keys:
                - "coords": [B, N, 3] normalized patch center coordinates.
                - "grid_size": [3] tensor with (D_out, H_out, W_out).
                - "patch_size": [3] tensor with (kD, kH, kW).
                - "volume_shape": [3] tensor with (D', H', W') of the feature map.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input of shape [B, C, D, H, W], got {tuple(x.shape)}")

        # Step 1: 3D encoder φ(x; θ_s)
        f0 = self.encoder(x)  # [B, C_stem, D', H', W']
        B, C, D, H, W = f0.shape

        # Step 2: Lazy initialization of projections W_t and g_aux.
        self._lazy_init_linears(f0)
        assert self.token_proj is not None
        assert self.aux_proj is not None

        # Step 3: Patch extraction ψ(f0, Ω_i)
        patch_size = self.cfg.patch_size
        stride = self.cfg.patch_stride

        patches, coords = extract_patches_3d(f0, patch_size, stride)  # [B, N, C*K], [B, N, 3]
        B, N, _ = patches.shape

        # Step 4: Token embeddings t_i = W_t ψ(f0, Ω_i) + b_t
        tokens = self.token_proj(patches)  # [B, N, d_token]
        tokens = self.token_norm(tokens)

        # Step 5: Auxiliary routing features a_i = g_aux(f0[Ω_i])
        # Here implemented as a lightweight linear projection of patch features.
        # You can replace this with a dedicated small Conv3D network if desired.
        aux = self.aux_proj(patches)  # [B, N, d_aux]

        # Metadata for decoders / positional reconstruction.
        out_D, out_H, out_W = _compute_output_grid(D, H, W, patch_size, stride)
        meta: Dict[str, Tensor] = {
            "coords": coords,  # [B, N, 3] normalized centers
            "grid_size": torch.tensor([out_D, out_H, out_W], device=x.device, dtype=torch.long),
            "patch_size": torch.tensor(list(patch_size), device=x.device, dtype=torch.long),
            "volume_shape": torch.tensor([D, H, W], device=x.device, dtype=torch.long),
        }

        return tokens, aux, meta
