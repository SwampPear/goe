"""Vesuvius Challenge Surface Detection dataset for the Graph-of-Experts model.

Loads 3D CT scan volumes and binary surface masks from the Vesuvius Challenge
competition data, extracts overlapping 3D patches, normalizes intensities,
applies data augmentations, and returns PyTorch tensors ready for training.

Data Format Assumptions
-----------------------
The competition provides 3D CT chunks of carbonized Herculaneum scrolls scanned
at ESRF (BM18) and DLS (I12) synchrotrons.  Each chunk has a corresponding
binary mask labeling the papyrus sheet surfaces.

Supported volume file formats:
    .tif / .tiff  -- single multi-page TIFF (3D volume)
    directory     -- numbered .tif slices stacked along depth axis
    .npy          -- NumPy binary
    .npz          -- compressed NumPy (first array or 'arr_0')
    .zarr         -- Zarr array
    .nrrd         -- NRRD (requires ``pip install pynrrd``)

Primary layout (CSV index):
    data_root/
    +-- index.csv              # volume manifest
    +-- volumes/               # CT scan chunks
    |   +-- chunk_001.tif
    |   +-- chunk_002.npy
    +-- surfaces/              # binary surface masks (same shape as volume)
        +-- chunk_001.tif
        +-- chunk_002.npy

    index.csv columns (required):
        id, volume_path, split
    index.csv columns (optional):
        surface_path, spacing_z, spacing_y, spacing_x, spacing

    Paths are relative to data_root.

Fallback layout (auto-discovery, no CSV):
    data_root/
    +-- train/
    |   +-- <image_id>/
    |       +-- volume.tif     (or volume.npy, or directory of .tif slices)
    |       +-- mask.tif       (or mask.npy; absent for test)
    +-- test/
        +-- <image_id>/
            +-- volume.tif
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Array I/O
# ---------------------------------------------------------------------------


def load_array(path: Path) -> np.ndarray:
    """Load a 3D (or 4D) array from disk. Supports .npy, .npz, .tif/.tiff (single file or directory of slices),
    .zarr, and .nrrd.  For directories the slices are stacked along the first (depth) axis in lexicographic order.
    """
    path = Path(path)

    # Directory of .tif slices
    if path.is_dir():
        return _load_tif_stack_dir(path)

    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(str(path), mmap_mode="r")

    if suffix == ".npz":
        with np.load(str(path)) as z:
            key = "arr_0" if "arr_0" in z else list(z.keys())[0]
            return z[key]

    if suffix in (".tif", ".tiff"):
        import tifffile

        return tifffile.imread(str(path))

    if suffix == ".zarr":
        import zarr

        return np.asarray(zarr.open(str(path), mode="r"))

    if suffix == ".nrrd":
        try:
            import nrrd
        except ImportError:
            raise ImportError(
                "pynrrd is required to read .nrrd files: pip install pynrrd"
            )
        data, _ = nrrd.read(str(path))
        return data

    raise ValueError(f"Unsupported file format: {suffix} ({path})")


def _load_tif_stack_dir(dirpath: Path) -> np.ndarray:
    """Load a directory of numbered .tif slices as a 3D volume (D, H, W)."""
    import tifffile

    tif_files = sorted(
        [f for f in dirpath.iterdir() if f.suffix.lower() in (".tif", ".tiff")]
    )
    if not tif_files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {dirpath}")

    slices = [tifffile.imread(str(f)) for f in tif_files]
    return np.stack(slices, axis=0)


def _get_volume_shape(path: Path) -> Tuple[int, ...]:
    """Read volume spatial shape without fully loading into memory.

    Returns the shape tuple.  Falls back to a full load when metadata
    inspection is not possible for the format.
    """
    path = Path(path)

    if path.is_dir():
        import tifffile

        tifs = sorted(
            [f for f in path.iterdir() if f.suffix.lower() in (".tif", ".tiff")]
        )
        if not tifs:
            raise FileNotFoundError(f"No .tif files in {path}")
        first = tifffile.imread(str(tifs[0]))
        return (len(tifs), *first.shape)

    suffix = path.suffix.lower()

    if suffix == ".npy":
        # mmap only reads the header; shape is immediate.
        arr = np.load(str(path), mmap_mode="r")
        shape = arr.shape
        del arr
        return shape

    if suffix in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            return tif.series[0].shape

    if suffix == ".zarr":
        import zarr

        return zarr.open(str(path), mode="r").shape

    # Fallback: full load.
    return load_array(path).shape


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_volume(
    volume: np.ndarray,
    method: str = "percentile",
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> np.ndarray:
    """Normalize CT intensity values to a standard range.

    Args:
        volume: 3D array (D, H, W).
        method: Normalization strategy.
            ``"percentile"`` -- clip to [p_low, p_high] then scale to [0, 1].
            ``"minmax"``     -- scale full range to [0, 1].
            ``"zscore"``     -- zero-mean, unit-variance.
            ``"fixed"``      -- ``(v - mean) / std`` using provided values.
            ``"raw"``        -- no-op.
        percentile_low: Lower percentile for ``"percentile"`` mode.
        percentile_high: Upper percentile for ``"percentile"`` mode.
        mean: Required when ``method="fixed"``.
        std: Required when ``method="fixed"``.

    Returns:
        float32 array of the same shape.
    """
    volume = volume.astype(np.float32)

    if method == "raw":
        return volume

    if method == "minmax":
        vmin, vmax = float(volume.min()), float(volume.max())
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)
        return volume

    if method == "percentile":
        vmin = float(np.percentile(volume, percentile_low))
        vmax = float(np.percentile(volume, percentile_high))
        volume = np.clip(volume, vmin, vmax)
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)
        return volume

    if method == "zscore":
        mu, sigma = float(volume.mean()), float(volume.std())
        if sigma > 0:
            volume = (volume - mu) / sigma
        return volume

    if method == "fixed":
        if mean is None or std is None:
            raise ValueError("method='fixed' requires mean and std")
        if std > 0:
            volume = (volume - mean) / std
        return volume

    raise ValueError(f"Unknown normalization method: {method!r}")


# ---------------------------------------------------------------------------
# Patch grid
# ---------------------------------------------------------------------------


@dataclass
class PatchGrid:
    """Regular grid of 3D patch coordinates over a volume.

    Attributes:
        starts: Patch start coordinates, shape ``(N, 3)`` as ``(z, y, x)``.
        patch_size: Patch dimensions ``(Dz, Dy, Dx)``.
        volume_shape: Source volume spatial shape ``(D, H, W)``.
    """

    starts: Tensor
    patch_size: Tuple[int, int, int]
    volume_shape: Tuple[int, int, int]

    @property
    def num_patches(self) -> int:
        return int(self.starts.shape[0])


def compute_patch_grid(
    volume_shape: Sequence[int],
    patch_size: Sequence[int],
    stride: Sequence[int],
) -> PatchGrid:
    """Build a regular grid of overlapping 3D patches.

    Uses *valid* coverage: patches stay within volume bounds, no padding.
    Choose ``stride < patch_size`` for overlapping patches.

    Args:
        volume_shape: ``(D, H, W)`` of the source volume.
        patch_size: ``(Dz, Dy, Dx)`` size of each patch.
        stride: ``(Sz, Sy, Sx)`` step between patch starts.

    Returns:
        PatchGrid with ``(N, 3)`` start coordinates.
    """
    D, H, W = int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2])
    Dz, Dy, Dx = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    Sz, Sy, Sx = int(stride[0]), int(stride[1]), int(stride[2])

    if any(p <= 0 for p in (Dz, Dy, Dx)):
        raise ValueError(f"patch_size must be positive, got ({Dz}, {Dy}, {Dx})")
    if any(s <= 0 for s in (Sz, Sy, Sx)):
        raise ValueError(f"stride must be positive, got ({Sz}, {Sy}, {Sx})")
    if Dz > D or Dy > H or Dx > W:
        raise ValueError(
            f"patch_size ({Dz},{Dy},{Dx}) exceeds volume_shape ({D},{H},{W})"
        )

    starts = [
        (z, y, x)
        for z in range(0, D - Dz + 1, Sz)
        for y in range(0, H - Dy + 1, Sy)
        for x in range(0, W - Dx + 1, Sx)
    ]

    if not starts:
        raise RuntimeError(
            f"No patches for volume ({D},{H},{W}) with "
            f"patch_size ({Dz},{Dy},{Dx}) stride ({Sz},{Sy},{Sx})"
        )

    return PatchGrid(
        starts=torch.tensor(starts, dtype=torch.long),
        patch_size=(Dz, Dy, Dx),
        volume_shape=(D, H, W),
    )


# ---------------------------------------------------------------------------
# 3D data augmentations
# ---------------------------------------------------------------------------

# Transforms operate on sample dicts with 5-D tensors (B, C, D, H, W).
# Spatial transforms are applied identically to all keys in _SPATIAL_KEYS.

_SPATIAL_KEYS = ("volume", "mask")


class RandomFlip3D:
    """Independent random flips along D, H, W axes.

    Each axis is flipped with probability ``p``.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        dims_to_flip = [d for d in (2, 3, 4) if torch.rand(()).item() < self.p]
        if not dims_to_flip:
            return sample
        for k in _SPATIAL_KEYS:
            if k in sample and sample[k].dim() == 5:
                sample[k] = torch.flip(sample[k], dims=dims_to_flip)
        return sample


class RandomRotate90:
    """Random 90-degree rotation in the H x W plane.

    With probability ``p``, rotates by k*90 degrees where k in {1, 2, 3}.

    Note: if patch H != W the spatial dimensions will swap on 90/270
    degree rotations.  Use square patches to avoid shape changes.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if torch.rand(()).item() >= self.p:
            return sample
        k = int(torch.randint(1, 4, (1,)).item())
        for key in _SPATIAL_KEYS:
            if key not in sample or sample[key].dim() != 5:
                continue
            x = sample[key]
            B, C, D, H, W = x.shape
            x = x.reshape(B * C * D, H, W)
            x = torch.rot90(x, k=k, dims=(-2, -1))
            sample[key] = x.reshape(B, C, D, x.shape[-2], x.shape[-1])
        return sample


class IntensityJitter:
    """Random multiplicative gain and additive bias on the volume.

    ``volume := volume * g + b`` where g ~ U(gain_range), b ~ U(bias_range).
    Applied only to the ``"volume"`` key.
    """

    def __init__(
        self,
        gain_range: Tuple[float, float] = (0.9, 1.1),
        bias_range: Tuple[float, float] = (-0.05, 0.05),
        p: float = 0.8,
    ) -> None:
        self.gain_range = gain_range
        self.bias_range = bias_range
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "volume" not in sample or torch.rand(()).item() >= self.p:
            return sample
        g = torch.empty(()).uniform_(*self.gain_range)
        b = torch.empty(()).uniform_(*self.bias_range)
        sample["volume"] = sample["volume"] * g + b
        return sample


class GaussianNoise:
    """Additive Gaussian noise on the volume.

    ``volume := volume + N(0, sigma^2)``.
    Applied only to the ``"volume"`` key.
    """

    def __init__(self, sigma: float = 0.02, p: float = 0.5) -> None:
        self.sigma = sigma
        self.p = p

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "volume" not in sample or torch.rand(()).item() >= self.p:
            return sample
        sample["volume"] = sample["volume"] + torch.randn_like(sample["volume"]) * self.sigma
        return sample


class ElasticDeformation3D:
    """Random elastic deformation via smooth displacement fields.

    Generates a random displacement field, smooths it with a Gaussian
    kernel, and warps the volume/mask.  The mask uses nearest-neighbor
    interpolation to stay binary.

    Requires ``scipy``.  If scipy is not installed the transform is
    silently skipped.
    """

    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 5.0,
        p: float = 0.2,
    ) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self._available: bool
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates  # noqa: F401

            self._available = True
        except ImportError:
            self._available = False
            logger.warning(
                "scipy not installed; ElasticDeformation3D will be skipped. "
                "Install with: pip install scipy"
            )

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self._available or torch.rand(()).item() >= self.p:
            return sample

        from scipy.ndimage import gaussian_filter, map_coordinates

        vol = sample.get("volume")
        if vol is None or vol.dim() != 5:
            return sample

        _, _, D, H, W = vol.shape

        # Random displacement field smoothed by Gaussian.
        dz = gaussian_filter(np.random.randn(D, H, W) * self.alpha, self.sigma)
        dy = gaussian_filter(np.random.randn(D, H, W) * self.alpha, self.sigma)
        dx = gaussian_filter(np.random.randn(D, H, W) * self.alpha, self.sigma)

        z, y, x = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing="ij"
        )
        coords = [z + dz, y + dy, x + dx]

        for k in _SPATIAL_KEYS:
            if k not in sample or sample[k].dim() != 5:
                continue
            t = sample[k]
            B, C = t.shape[:2]
            arr = t.numpy()
            out = np.empty_like(arr)
            order = 1 if k == "volume" else 0  # nearest for masks
            for b in range(B):
                for c in range(C):
                    out[b, c] = map_coordinates(
                        arr[b, c], coords, order=order, mode="reflect"
                    )
            sample[k] = torch.from_numpy(out)

        return sample


def default_augmentations(include_elastic: bool = False) -> List[Callable]:
    """Build the default augmentation pipeline.

    Args:
        include_elastic: Include elastic deformation (requires scipy, slow).

    Returns:
        List of augmentation callables that operate on sample dicts.
    """
    augs: List[Callable] = [
        RandomFlip3D(p=0.5),
        RandomRotate90(p=0.5),
        IntensityJitter(gain_range=(0.9, 1.1), bias_range=(-0.05, 0.05), p=0.8),
        GaussianNoise(sigma=0.02, p=0.5),
    ]
    if include_elastic:
        augs.append(ElasticDeformation3D(alpha=50.0, sigma=5.0, p=0.2))
    return augs


# ---------------------------------------------------------------------------
# Volume index
# ---------------------------------------------------------------------------


@dataclass
class VolumeEntry:
    """Metadata for a single volume in the dataset."""

    volume_id: str
    volume_path: Path
    surface_path: Optional[Path] = None
    spacing: Optional[Tuple[float, float, float]] = None
    split: str = "train"


def _parse_spacing(row: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    """Parse voxel spacing from a CSV row.

    Supports ``spacing_z, spacing_y, spacing_x`` columns or a single
    ``spacing`` column formatted as ``"z,y,x"`` or ``"z y x"``.
    """
    if all(k in row and row[k] for k in ("spacing_z", "spacing_y", "spacing_x")):
        return (
            float(row["spacing_z"]),
            float(row["spacing_y"]),
            float(row["spacing_x"]),
        )

    if "spacing" in row and row["spacing"]:
        raw = row["spacing"].replace(",", " ").strip()
        parts = raw.split()
        if len(parts) == 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))

    return None


def load_index(root: Path) -> List[VolumeEntry]:
    """Load the volume manifest from ``root/index.csv``.

    Required columns: ``id``, ``volume_path``, ``split``.
    Optional columns: ``surface_path``, ``spacing_z``, ``spacing_y``,
    ``spacing_x``, ``spacing``.
    """
    index_path = root / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing index.csv at {index_path}. "
            f"Create one with columns: id,volume_path,surface_path,split"
        )

    entries: List[VolumeEntry] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            volume_path = (root / row["volume_path"]).resolve()
            surface_path = (
                (root / row["surface_path"]).resolve()
                if row.get("surface_path")
                else None
            )
            entries.append(
                VolumeEntry(
                    volume_id=row["id"],
                    volume_path=volume_path,
                    surface_path=surface_path,
                    spacing=_parse_spacing(row),
                    split=row.get("split", "train"),
                )
            )
    return entries


def discover_volumes(root: Path, split: str = "train") -> List[VolumeEntry]:
    """Auto-discover volumes when no ``index.csv`` is present.

    Scans ``root/<split>/`` for sub-directories, each expected to contain
    a volume file (``volume.tif``, ``volume.npy``, a ``surface_volume/``
    directory of slices, or any lone ``.tif``/``.npy`` file) and optionally
    a mask file (``mask.tif``, ``mask.npy``, ``surface.tif``).
    """
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    entries: List[VolumeEntry] = []
    for vol_dir in sorted(split_dir.iterdir()):
        if not vol_dir.is_dir():
            continue

        volume_path = _find_file(
            vol_dir,
            named=("volume.tif", "volume.tiff", "volume.npy", "volume.npz"),
            subdir="surface_volume",
        )
        if volume_path is None:
            logger.warning(f"No volume found in {vol_dir}, skipping")
            continue

        mask_path = _find_file(
            vol_dir,
            named=(
                "mask.tif", "mask.tiff", "mask.npy", "mask.npz",
                "surface.tif", "surface.tiff", "surface.npy",
            ),
        )

        entries.append(
            VolumeEntry(
                volume_id=vol_dir.name,
                volume_path=volume_path,
                surface_path=mask_path,
                split=split,
            )
        )

    return entries


def _find_file(
    dirpath: Path,
    named: Sequence[str] = (),
    subdir: Optional[str] = None,
) -> Optional[Path]:
    """Locate a file inside *dirpath* by checking known names, then a
    sub-directory of slices, then any lone matching file."""
    for name in named:
        p = dirpath / name
        if p.exists():
            return p
    if subdir is not None:
        d = dirpath / subdir
        if d.is_dir():
            return d
    # Lone file fallback.
    for ext in (".tif", ".tiff", ".npy", ".npz", ".nrrd"):
        matches = list(dirpath.glob(f"*{ext}"))
        if len(matches) == 1:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VesuviusPatchDataset(Dataset):
    """Patch-based PyTorch dataset for Vesuvius surface detection.

    Divides each volume into overlapping 3D patches on a regular grid.
    Each ``__getitem__`` call returns **one patch**; the ``DataLoader``
    batches them into ``(B, 1, Dz, Dy, Dx)`` tensors for the model.

    Args:
        root: Data root directory (contains ``index.csv`` or ``train/``).
        split: Data split -- ``"train"``, ``"val"``, or ``"test"``.
        patch_size: ``(D, H, W)`` dimensions of each 3D patch.
        stride: ``(D, H, W)`` step between adjacent patches.
            Use ``stride < patch_size`` for overlapping coverage.
        normalize: Intensity normalization method (see
            :func:`normalize_volume`).  ``"percentile"`` works well for
            CT data with varying scanner intensity ranges.
        augment: Apply data augmentations during training.
        augmentations: Custom augmentation list.  When ``None`` and
            ``augment=True``, :func:`default_augmentations` is used.
        allow_missing_masks: If ``False``, raises when a volume has no
            associated surface mask (set ``True`` for test data).
        limit_volumes: Only load this many volumes (for debugging).
        cache_volumes: Keep the most-recently-loaded volume in memory
            to avoid repeated disk reads when sampling multiple patches
            from the same volume.

    Per-sample return dict:
        ``"volume"``      -- ``(1, Dz, Dy, Dx)`` float32 CT patch.
        ``"mask"``        -- ``(Dz, Dy, Dx)`` float32 binary surface mask
                             (omitted for test data without labels).
        ``"volume_id"``   -- ``str`` source volume identifier.
        ``"patch_start"`` -- ``(3,)`` long tensor ``(z, y, x)`` start coords.
        ``"spacing"``     -- ``(3,)`` float tensor if available.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        patch_size: Sequence[int] = (64, 64, 64),
        stride: Sequence[int] = (32, 32, 32),
        normalize: str = "percentile",
        augment: bool = False,
        augmentations: Optional[List[Callable]] = None,
        allow_missing_masks: bool = True,
        limit_volumes: Optional[int] = None,
        cache_volumes: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.patch_size = tuple(int(x) for x in patch_size)
        self.stride = tuple(int(x) for x in stride)
        self.normalize = normalize
        self.allow_missing_masks = allow_missing_masks
        self.cache_volumes = cache_volumes

        # ---- load volume manifest ----
        if (self.root / "index.csv").exists():
            all_entries = load_index(self.root)
            self.entries = [e for e in all_entries if e.split == split]
        else:
            self.entries = discover_volumes(self.root, split)

        if limit_volumes is not None:
            self.entries = self.entries[: int(limit_volumes)]

        if not self.entries:
            raise RuntimeError(
                f"No volumes found for split='{split}' in {self.root}. "
                f"Ensure index.csv exists or data follows the expected "
                f"directory layout (see module docstring)."
            )

        logger.info(
            f"[{split}] {len(self.entries)} volume(s) from {self.root}"
        )

        # ---- build per-volume patch grids ----
        self._grids: List[PatchGrid] = []
        self._flat_index: List[Tuple[int, int]] = []  # (entry_idx, patch_idx)

        for entry_idx, entry in enumerate(self.entries):
            shape = _get_volume_shape(entry.volume_path)
            # Use last 3 dims as (D, H, W) regardless of channel dim.
            spatial = shape[-3:] if len(shape) >= 3 else shape
            grid = compute_patch_grid(spatial, self.patch_size, self.stride)
            self._grids.append(grid)
            for p_idx in range(grid.num_patches):
                self._flat_index.append((entry_idx, p_idx))

        logger.info(
            f"[{split}] {len(self._flat_index)} patches "
            f"(patch_size={self.patch_size}, stride={self.stride})"
        )

        # ---- volume cache (one volume at a time) ----
        self._cache_entry: Optional[int] = None
        self._cache_vol: Optional[np.ndarray] = None
        self._cache_mask: Optional[np.ndarray] = None

        # ---- augmentations ----
        if augment:
            self._augmentations = augmentations or default_augmentations()
        else:
            self._augmentations: List[Callable] = []

    # ----- internal helpers ------------------------------------------------

    def _load_entry(
        self, entry_idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load (and optionally cache) a volume and its mask."""
        if self.cache_volumes and self._cache_entry == entry_idx:
            assert self._cache_vol is not None
            return self._cache_vol, self._cache_mask

        entry = self.entries[entry_idx]

        # -- volume --
        vol = np.array(load_array(entry.volume_path))
        if vol.ndim == 4:
            vol = vol[0]  # drop leading channel dim, keep (D, H, W)
        if vol.ndim != 3:
            raise ValueError(
                f"Expected 3D volume after squeeze, got shape {vol.shape} "
                f"for '{entry.volume_id}'"
            )
        vol = normalize_volume(vol, method=self.normalize)

        # -- mask --
        mask: Optional[np.ndarray] = None
        if entry.surface_path is not None:
            mask = np.array(load_array(entry.surface_path)).astype(np.float32)
            if mask.ndim == 4:
                mask = mask[0]
            # Binarize: handle masks saved as uint8 255 / uint16 / etc.
            if mask.max() > 1.0:
                mask = (mask > 0.5 * mask.max()).astype(np.float32)
        elif not self.allow_missing_masks:
            raise RuntimeError(
                f"No surface mask for volume '{entry.volume_id}'. "
                f"Set allow_missing_masks=True for test data."
            )

        if self.cache_volumes:
            self._cache_entry = entry_idx
            self._cache_vol = vol
            self._cache_mask = mask

        return vol, mask

    # ----- Dataset interface -----------------------------------------------

    def __len__(self) -> int:
        return len(self._flat_index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry_idx, patch_idx = self._flat_index[idx]
        entry = self.entries[entry_idx]
        grid = self._grids[entry_idx]

        start = grid.starts[patch_idx].tolist()
        z0, y0, x0 = int(start[0]), int(start[1]), int(start[2])
        Dz, Dy, Dx = grid.patch_size

        vol, mask = self._load_entry(entry_idx)

        # -- extract patch --
        vol_patch = vol[z0 : z0 + Dz, y0 : y0 + Dy, x0 : x0 + Dx].copy()
        vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0)  # (1, Dz, Dy, Dx)

        sample: Dict[str, Any] = {"volume": vol_tensor}

        if mask is not None:
            mask_patch = mask[z0 : z0 + Dz, y0 : y0 + Dy, x0 : x0 + Dx].copy()
            sample["mask"] = torch.from_numpy(mask_patch)  # (Dz, Dy, Dx)

        # -- augmentations (need batch dim for 5-D transforms) --
        if self._augmentations:
            sample["volume"] = sample["volume"].unsqueeze(0)  # (1,1,D,H,W)
            if "mask" in sample:
                sample["mask"] = sample["mask"].unsqueeze(0).unsqueeze(0)

            for aug in self._augmentations:
                sample = aug(sample)

            sample["volume"] = sample["volume"].squeeze(0)  # (1,D,H,W)
            if "mask" in sample:
                sample["mask"] = sample["mask"].squeeze(0).squeeze(0)

        # -- metadata --
        sample["volume_id"] = entry.volume_id
        sample["patch_start"] = torch.tensor([z0, y0, x0], dtype=torch.long)
        if entry.spacing is not None:
            sample["spacing"] = torch.tensor(entry.spacing, dtype=torch.float32)

        return sample


# ---------------------------------------------------------------------------
# Collation and DataLoader helpers
# ---------------------------------------------------------------------------


def collate_patches(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate for :class:`VesuviusPatchDataset` samples.

    Stacks tensor fields, collects string metadata into lists, and
    gracefully handles optional keys (``"mask"``, ``"spacing"``).
    """
    out: Dict[str, Any] = {
        "volume": torch.stack([b["volume"] for b in batch]),
    }

    if "mask" in batch[0]:
        out["mask"] = torch.stack([b["mask"] for b in batch])

    out["volume_id"] = [b["volume_id"] for b in batch]
    out["patch_start"] = torch.stack([b["patch_start"] for b in batch])

    if "spacing" in batch[0]:
        out["spacing"] = torch.stack([b["spacing"] for b in batch])

    return out


def create_dataloaders(
    root: Union[str, Path],
    patch_size: Sequence[int] = (64, 64, 64),
    stride: Sequence[int] = (32, 32, 32),
    batch_size: int = 4,
    num_workers: int = 0,
    normalize: str = "percentile",
    **dataset_kwargs: Any,
) -> Dict[str, DataLoader]:
    """Convenience factory for train and val ``DataLoader`` instances.

    Args:
        root: Data root directory.
        patch_size: 3D patch dimensions.
        stride: Patch stride.
        batch_size: Batch size.
        num_workers: Number of DataLoader worker processes.
        normalize: Normalization method.
        **dataset_kwargs: Forwarded to :class:`VesuviusPatchDataset`.

    Returns:
        Dict with ``"train"`` and/or ``"val"`` keys mapping to
        DataLoader instances.  A split is silently skipped when no
        data is found for it.
    """
    loaders: Dict[str, DataLoader] = {}

    for split, augment, shuffle in [("train", True, True), ("val", False, False)]:
        try:
            ds = VesuviusPatchDataset(
                root=root,
                split=split,
                patch_size=patch_size,
                stride=stride,
                normalize=normalize,
                augment=augment,
                **dataset_kwargs,
            )
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_patches,
            )
        except (RuntimeError, FileNotFoundError) as e:
            logger.warning(f"Skipping {split} split: {e}")

    return loaders


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Create synthetic data and exercise the full pipeline."""
    import tempfile

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("VesuviusPatchDataset -- synthetic demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        vol_dir = root / "volumes"
        mask_dir = root / "surfaces"
        vol_dir.mkdir()
        mask_dir.mkdir()

        # -- create a synthetic 3D volume and surface mask --
        D, H, W = 128, 128, 128
        volume = np.random.randn(D, H, W).astype(np.float32) * 100 + 500

        # Simulate a curved papyrus sheet surface.
        mask = np.zeros((D, H, W), dtype=np.float32)
        zz, yy = np.meshgrid(np.arange(D), np.arange(H), indexing="ij")
        surface_x = (
            W // 2
            + (10 * np.sin(2 * np.pi * zz / D))
            + (5 * np.cos(2 * np.pi * yy / H))
        ).astype(int)
        for offset in (-1, 0, 1):
            sx = np.clip(surface_x + offset, 0, W - 1)
            mask[zz, yy, sx] = 1.0

        np.save(vol_dir / "demo_vol.npy", volume)
        np.save(mask_dir / "demo_vol.npy", mask)

        # -- write index.csv --
        with open(root / "index.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "volume_path", "surface_path", "split"])
            writer.writerow(
                ["demo_vol", "volumes/demo_vol.npy", "surfaces/demo_vol.npy", "train"]
            )

        # -- create dataset --
        ds = VesuviusPatchDataset(
            root=root,
            split="train",
            patch_size=(32, 32, 32),
            stride=(16, 16, 16),
            normalize="percentile",
            augment=True,
        )

        print(f"\nPatches total : {len(ds)}")

        # -- inspect one sample --
        sample = ds[0]
        print(f"volume shape  : {sample['volume'].shape}")
        print(f"volume range  : [{sample['volume'].min():.3f}, {sample['volume'].max():.3f}]")
        if "mask" in sample:
            print(f"mask shape    : {sample['mask'].shape}")
            print(f"mask unique   : {sample['mask'].unique().tolist()}")
        print(f"volume_id     : {sample['volume_id']}")
        print(f"patch_start   : {sample['patch_start'].tolist()}")

        # -- test DataLoader with custom collate --
        loader = DataLoader(
            ds, batch_size=4, shuffle=True, collate_fn=collate_patches
        )
        batch = next(iter(loader))
        print(f"\nbatch volume  : {batch['volume'].shape}")
        if "mask" in batch:
            print(f"batch mask    : {batch['mask'].shape}")
        print(f"batch ids     : {batch['volume_id']}")
        print(f"batch starts  : {batch['patch_start'].shape}")

        # -- test create_dataloaders helper --
        print("\n-- create_dataloaders (train only, val will be skipped) --")
        loaders = create_dataloaders(
            root=root,
            patch_size=(32, 32, 32),
            stride=(16, 16, 16),
            batch_size=2,
        )
        for name, ldr in loaders.items():
            print(f"  {name}: {len(ldr)} batches, {len(ldr.dataset)} patches")

    print("\nDemo complete.")
