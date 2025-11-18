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

    Attributes:
        stem_channels: # of feature channels produced by the 3D encoder
        token_dim: dimensionality of token embeddings consumed by graph router and experts
        aux_dim: dimensionality of auxiliary routing features used by policy modules
        patch_size: size of each 3D patch (kD, kH, kW) in voxels defining receptive field of each token
        patch_stride: stride between consecutive patches (sD, sH, sW)
        use_layer_norm: whether to apply LayerNorm to token embeddings before feeding to router
    """
    stem_channels: int = 32
    token_dim: int = 128
    aux_dim: int = 32
    patch_size: Tuple[int, int, int] = (16, 16, 16)
    patch_stride: Tuple[int, int, int] = (8, 8, 8)
    use_layer_norm: bool = True


class Conv3DEncoder(nn.Module):
    """
    Lightweight 3D convolutional encoder for extracting localized features such as carbonization gradients, fiber 
    structure, and ink-related density anomalies from the raw volume. Early layers should capture fine-scale surface
    features. A final stride=2 layer reduces spatial resolution while increasing channel width, producing a compact 
    feature grid used for later patch-tokenization.
    """

    def __init__(self, stem_channels: int):
        super().__init__()

        mid = stem_channels // 2

        self.encoder = nn.Sequential(
            # first feature extractor block
            nn.Conv3d(1, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),

            # second refinement block (increases local contrast sensitivity)
            nn.Conv3d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),

            # downsampling block: expands to full stem_channels
            nn.Conv3d(mid, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the 3D encoder.
        Args:
            x: Tensor of shape [B, C_in, D, H, W] — raw tomographic volume.
        Returns:
            Tensor of shape [B, C_stem, D', H', W'] —
            A lower-resolution feature grid summarizing local ink/fiber cues.
        """
        return self.encoder(x)


def _compute_output_grid(
    D: int, H: int, W: int, 
    patch_size: Tuple[int, int, int], 
    stride: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
    """
    Computes how many patches fit along each spatial dimension after sliding a 3D window of size (kD, kH, kW) across a 
    volume of size (D, H, W). Used for determining token grid shape, allocating positional encodings, and ensuring patch
    extraction loops are dimensionally correct.

    The formula matches PyTorch-style sliding window semantics:
        out = floor((size - kernel) / stride) + 1
    """
    kD, kH, kW = patch_size
    sD, sH, sW = stride

    out_D = math.floor((D - kD) / sD) + 1
    out_H = math.floor((H - kH) / sH) + 1
    out_W = math.floor((W - kW) / sW) + 1

    return out_D, out_H, out_W


def extract_patches_3d(
    features: Tensor,
    patch_size: Tuple[int, int, int], 
    stride: Tuple[int, int, int]
    ) -> Tuple[Tensor, Tensor]:
    """
    Extract overlapping 3D patches from a feature volume. Gathers local voxel neighborhoods around each position.
    Args:
        features: Tensor of shape [B, C, D, H, W] - output of convolutional encoder
        patch_size: (kD, kH, kW) - spatial extent of each extracted 3D patch
        stride: (sD, sH, sW) - spatial step size between patch locations.
    Returns:
        (patches, coords)
        patches: Tensor of shape [B, N, C * kD * kH * kW] where N = D_out * H_out * W_out and each entry is a flattened
            voxel cube capturing local structure
        coords: Tensor of shape [B, N, 3] giving the normalized centers of each patch along (z, y, x) used for 
            positional encodings and routing heuristics
    """
    B, C, D, H, W = features.shape
    kD, kH, kW = patch_size
    sD, sH, sW = stride

    # extract sliding windows along a dimension where (kD,kH,kW) is the patch and (D_out,H_out,W_out) is the grid
    unfolded = (
        features
        .unfold(dimension=2, size=kD, step=sD)   # slide along depth
        .unfold(dimension=3, size=kH, step=sH)   # slide along height
        .unfold(dimension=4, size=kW, step=sW)   # slide along width
    )

    B, C, D_out, H_out, W_out, _, _, _ = unfolded.shape

    # flatten patches into [B, N, C * kD * kH * kW]
    # collapse spatial grid (D_out * H_out * W_out) into N
    patches = unfolded.contiguous().view(
        B, C, D_out * H_out * W_out, kD * kH * kW
    )

    # move channels next to patch voxels [B, N, C, K]
    patches = patches.permute(0, 2, 1, 3).contiguous()

    # flatten all patch voxels + channel dimension together
    patches = patches.view(B, D_out * H_out * W_out, C * kD * kH * kW)

    # compute the center of every patch in the feature-grid coordinate system.
    device = features.device

    # center = start_index + (patch_size-1)/2
    z_idxs = torch.arange(D_out, device=device) * sD + (kD - 1) / 2.0
    y_idxs = torch.arange(H_out, device=device) * sH + (kH - 1) / 2.0
    x_idxs = torch.arange(W_out, device=device) * sW + (kW - 1) / 2.0

    # meshgrid produces coordinates for the whole patch grid
    zz, yy, xx = torch.meshgrid(z_idxs, y_idxs, x_idxs, indexing="ij")

    # normalize by (size - 1) so the largest coordinate maps to 1.0
    zz = zz.reshape(-1) / max(D - 1, 1)
    yy = yy.reshape(-1) / max(H - 1, 1)
    xx = xx.reshape(-1) / max(W - 1, 1)

    coords = torch.stack([zz, yy, xx], dim=-1)   # [N, 3]
    coords = coords.unsqueeze(0).repeat(B, 1, 1) # broadcast batch dimension

    return patches, coords


class InputStem(nn.Module):
    """
    Input stem mapping raw 3D X-ray volumes into token embeddings and auxiliary routing features. After input, the
    output tokens and auxiliary features are consumed by the encoder and graph router.
    """
    def __init__(self, cfg: InputStemConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Conv3DEncoder(stem_channels=cfg.stem_channels)

        # projections
        self.token_proj: Optional[nn.Linear] = None # from flattened patch features to token embeddings
        self.aux_proj: Optional[nn.Linear] = None   # from flattened patch features to auxiliary routing features

        # LayerNorm over token_dim for stability.
        self.token_norm = nn.LayerNorm(cfg.token_dim) if cfg.use_layer_norm else nn.Identity()

        # lazily initialize linear layers once spacial dimensions are known after encoding to compute patch size
        self._initialized = False


    def _lazy_init_linears(self, f0: Tensor) -> None:
        """
        Initialize token and auxiliary projections after seeing the first feature map, so patch volume size is known as
        C * kD * kH * kW.
        """
        if self._initialized:
            return

        _, C, D, H, W = f0.shape
        kD, kH, kW = self.cfg.patch_size
        sD, sH, sW = self.cfg.patch_stride

        # compute number of patch positions along each dimension, required to validate patch configuration
        out_D, out_H, out_W = _compute_output_grid(D, H, W, self.cfg.patch_size, self.cfg.patch_stride)
        if out_D <= 0 or out_H <= 0 or out_W <= 0:
            raise ValueError(
                f"Patch configuration yields no patches: "
                f"feat_volume=({D}, {H}, {W}), patch_size={self.cfg.patch_size}, "
                f"stride={self.cfg.patch_stride}"
            )

        # # of scalar values per patch
        flattened_patch_dim = C * kD * kH * kW

        # initialize projections
        self.token_proj = nn.Linear(flattened_patch_dim, self.cfg.token_dim)
        self.aux_proj = nn.Linear(flattened_patch_dim, self.cfg.aux_dim)

        self._initialized = True


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Forward pass through the InputStem.

        Args:
            x: Tensor of shape [B, C_in, D, H, W] - raw 3D microCT volume
        Returns:
            (tokens, aux, meta)
            tokens: Tensor of shape [B, N, d_token] - dense token embeddings
            aux: Tensor of shape [B, N, d_aux] - auxiliary routing features
            {
                'coords': [B, N, 3] normalized patch centers,
                'grid_size': (D_out, H_out, W_out) number of patch positions,
                'patch_size': (kD, kH, kW) window size,
                'volume_shape': (D', H', W') spatial dimensions of encoded f0
            }
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input of shape [B, C, D, H, W], got {tuple(x.shape)}.")

        # encode raw volume
        f0 = self.encoder(x)  # [B, C_s, D', H', W']
        B, C, D, H, W = f0.shape

        # lazily initialize projections
        self._lazy_init_linears(f0)
        assert self.token_proj is not None
        assert self.aux_proj is not None

        # extract 3D patches
        patch_size = self.cfg.patch_size
        stride = self.cfg.patch_stride
        patches, coords = extract_patches_3d(f0, patch_size, stride) # [B, N, C*K], [B, N, 3]
        B, N, _ = patches.shape

        # token embeddings
        tokens = self.token_proj(patches) # [B, N, d_token]
        tokens = self.token_norm(tokens)

        # auxiliary routing features
        aux = self.aux_proj(patches) # [B, N, d_aux]

        # metadata for downstream models
        out_D, out_H, out_W = _compute_output_grid(D, H, W, patch_size, stride)
        meta: Dict[str, Tensor] = {
            "coords": coords,
            "grid_size": torch.tensor([out_D, out_H, out_W], device=x.device, dtype=torch.long),
            "patch_size": torch.tensor(list(patch_size), device=x.device, dtype=torch.long),
            "volume_shape": torch.tensor([D, H, W], device=x.device, dtype=torch.long),
        }

        return tokens, aux, meta