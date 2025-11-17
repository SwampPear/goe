from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor


__all__ = [
    "PatchGrid",
    "compute_patch_grid",
    "extract_patches",
    "extract_patches_from_volume",
    "tokens_to_coords",
]


@dataclass
class PatchGrid:
    """
    Discrete grid of voxel patches Ω_i over a 3D volume.

    Attributes
    ----------
    starts : Tensor
        Integer start coordinates for each patch, shape (N, 3) as (z, y, x).
    patch_size : Tuple[int, int, int]
        Patch size (Dz, Dy, Dx).
    volume_shape : Tuple[int, int, int]
        Underlying volume shape (D, H, W) that the grid was constructed for.
    """

    starts: Tensor
    patch_size: Tuple[int, int, int]
    volume_shape: Tuple[int, int, int]

    @property
    def num_patches(self) -> int:
        return int(self.starts.shape[0])

    def to(self, device: torch.device | str) -> "PatchGrid":
        """
        Convenience: move start indices to a device (useful for JIT / logging).

        Note: indices are typically consumed as Python ints for slicing, so
        this is mostly for bookkeeping / consistency.
        """
        return PatchGrid(
            starts=self.starts.to(device=device),
            patch_size=self.patch_size,
            volume_shape=self.volume_shape,
        )


def _as_tuple3(x: int | Sequence[int]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    if len(x) != 3:
        raise ValueError(f"Expected length-3 sequence, got {x}")
    return int(x[0]), int(x[1]), int(x[2])


def compute_patch_grid(
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    stride: Sequence[int],
) -> PatchGrid:
    """
    Construct a regular grid of non-padded 3D patches over a volume.

    This implements the ψ(·, Ω_i) partition in the paper: we discretize the
    3D volume into overlapping voxel patches Ω_i with a given patch size and
    stride, using a simple "valid" coverage rule (no padding).

    Parameters
    ----------
    volume_shape : (D, H, W)
        Shape of the underlying volume (depth, height, width).
    patch_size : (Dz, Dy, Dx)
        Voxel size of each patch.
    stride : (Sz, Sy, Sx)
        Stride between patch starts along each axis.

    Returns
    -------
    PatchGrid
        A PatchGrid with starts of shape (N, 3) and the given patch_size.
    """
    D, H, W = _as_tuple3(volume_shape)
    Dz, Dy, Dx = _as_tuple3(patch_size)
    Sz, Sy, Sx = _as_tuple3(stride)

    if Dz <= 0 or Dy <= 0 or Dx <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if Sz <= 0 or Sy <= 0 or Sx <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if Dz > D or Dy > H or Dx > W:
        raise ValueError(
            f"patch_size {patch_size} is larger than volume_shape {volume_shape}"
        )

    zs = list(range(0, D - Dz + 1, Sz))
    ys = list(range(0, H - Dy + 1, Sy))
    xs = list(range(0, W - Dx + 1, Sx))

    starts_list: List[Tuple[int, int, int]] = []
    for z in zs:
        for y in ys:
            for x in xs:
                starts_list.append((z, y, x))

    if not starts_list:
        raise RuntimeError("No patches generated; check volume_shape/patch_size/stride.")

    starts = torch.tensor(starts_list, dtype=torch.long)  # (N, 3)

    return PatchGrid(
        starts=starts,
        patch_size=(Dz, Dy, Dx),
        volume_shape=(D, H, W),
    )


def extract_patches(
    volume: Tensor,
    grid: PatchGrid,
) -> Tensor:
    """
    Extract voxel patches Ω_i from a 5D volume using a precomputed PatchGrid.

    This is the implementation of ψ(f₀, Ω_i) in Eq. (1):

        f₀ ∈ ℝ^{B×C×D×H×W}
        Ω_i := [z₀:z₁, y₀:y₁, x₀:x₁]
        ψ(f₀, Ω_i) := f₀[:, :, z₀:z₁, y₀:y₁, x₀:x₁]

    Parameters
    ----------
    volume : Tensor
        Input volume, shape (B, C, D, H, W).
    grid : PatchGrid
        Grid of patch start coordinates Ω_i and patch size.

    Returns
    -------
    Tensor
        Extracted patches, shape (B, N, C, Dz, Dy, Dx) where
        N = grid.num_patches, (Dz, Dy, Dx) = grid.patch_size.
    """
    if volume.dim() != 5:
        raise ValueError(
            f"volume must have shape (B, C, D, H, W), got {tuple(volume.shape)}"
        )

    B, C, D, H, W = volume.shape
    Dz, Dy, Dx = grid.patch_size
    D0, H0, W0 = grid.volume_shape

    if (D, H, W) != (D0, H0, W0):
        raise ValueError(
            f"volume spatial shape {(D, H, W)} does not match grid.volume_shape {grid.volume_shape}"
        )

    N = grid.num_patches
    patches = volume.new_empty((B, N, C, Dz, Dy, Dx))

    # Loop over patches; simple but explicit. Can be optimized later with
    # as_strided / unfold3d if needed.
    for i, (z, y, x) in enumerate(grid.starts.tolist()):
        z0, y0, x0 = int(z), int(y), int(x)
        z1, y1, x1 = z0 + Dz, y0 + Dy, x0 + Dx
        patches[:, i] = volume[:, :, z0:z1, y0:y1, x0:x1]

    return patches


def extract_patches_from_volume(
    volume: Tensor,
    patch_size: Sequence[int],
    stride: Sequence[int],
) -> Tuple[Tensor, PatchGrid]:
    """
    Convenience helper: build a PatchGrid and immediately extract patches.

    Parameters
    ----------
    volume : Tensor
        Input volume, shape (B, C, D, H, W).
    patch_size : (Dz, Dy, Dx)
        Voxel size of each patch.
    stride : (Sz, Sy, Sx)
        Stride between patch starts along each axis.

    Returns
    -------
    patches : Tensor
        Extracted patches, shape (B, N, C, Dz, Dy, Dx).
    grid : PatchGrid
        The underlying patch grid (for token indexing, aux features, etc.).
    """
    if volume.dim() != 5:
        raise ValueError(
            f"volume must have shape (B, C, D, H, W), got {tuple(volume.shape)}"
        )

    _, _, D, H, W = volume.shape
    grid = compute_patch_grid(
        volume_shape=(D, H, W),
        patch_size=patch_size,
        stride=stride,
    )
    patches = extract_patches(volume, grid)
    return patches, grid


def tokens_to_coords(
    grid: PatchGrid,
    token_indices: Tensor,
    center: bool = True,
) -> Tensor:
    """
    Map token indices i → 3D coordinates inside the original volume.

    This is useful for:
      - projecting predictions back into the scroll volume, or
      - logging / visualization of token locations.

    Parameters
    ----------
    grid : PatchGrid
        Grid used to generate the tokens.
    token_indices : Tensor
        1D or 2D tensor of integer token indices (e.g., shape (N,) or (B, N)).
    center : bool, default True
        If True, return the voxel center of each patch Ω_i.
        If False, return the patch start coordinate (z₀, y₀, x₀).

    Returns
    -------
    Tensor
        Coordinates of each token in (z, y, x) order.
        Shape matches token_indices.shape + (3,).
    """
    if token_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_indices must be an integer tensor")

    starts = grid.starts  # (N, 3)
    Dz, Dy, Dx = grid.patch_size

    flat_idx = token_indices.reshape(-1)
    if flat_idx.min() < 0 or flat_idx.max() >= starts.shape[0]:
        raise IndexError(
            f"token index out of range [0, {starts.shape[0] - 1}] in {flat_idx.tolist()}"
        )

    coords = starts[flat_idx]  # (K, 3)
    if center:
        # Add half patch size to get approximate center (float coords).
        offset = torch.tensor(
            [Dz / 2.0, Dy / 2.0, Dx / 2.0],
            device=coords.device,
            dtype=torch.float32,
        )
        coords = coords.to(torch.float32) + offset

    coords = coords.view(*token_indices.shape, 3)
    return coords
