from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.augmentations import VesuviusAugmentations, VesuviusAugConfig
from src.data.patches import PatchGrid, compute_patch_grid


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        with np.load(path) as z:
            # Use the first array if no explicit key is known.
            key = "arr_0" if "arr_0" in z else list(z.keys())[0]
            return z[key]
    if path.suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    raise ValueError(f"Unsupported array format: {path}")


def _ensure_4d(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 3:
        return volume[None, ...]
    if volume.ndim == 4:
        return volume
    raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")


def _slice_patch(arr: np.ndarray, start: Sequence[int], size: Sequence[int]) -> np.ndarray:
    z0, y0, x0 = (int(start[0]), int(start[1]), int(start[2]))
    dz, dy, dx = (int(size[0]), int(size[1]), int(size[2]))
    return arr[..., z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]


@dataclass
class IndexEntry:
    volume_id: str
    volume_path: Path
    surface_path: Optional[Path]
    ink_path: Optional[Path]
    geometry_path: Optional[Path]
    spacing: Optional[Tuple[float, float, float]]
    split: str


class VesuviusPatchDataset(Dataset):
    """
    Patch-based dataset for volumetric Vesuvius data.

    Expected processed layout:
      root/
        index.csv
        volumes/*.npy
        ink/*.npy (optional)
        geometry/*.npy (optional)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        patch_size: Sequence[int] = (16, 64, 64),
        stride: Sequence[int] = (8, 32, 32),
        augment: bool = False,
        allow_missing_targets: bool = True,
        limit_volumes: Optional[int] = None,
        aug_cfg: Optional[VesuviusAugConfig] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.patch_size = tuple(int(x) for x in patch_size)
        self.stride = tuple(int(x) for x in stride)
        self.allow_missing_targets = allow_missing_targets

        self.entries = self._load_index()
        if split:
            self.entries = [e for e in self.entries if e.split == split]
        if limit_volumes is not None:
            self.entries = self.entries[: int(limit_volumes)]

        if not self.entries:
            raise RuntimeError(f"No dataset entries found for split='{split}'.")

        self._grids: List[PatchGrid] = []
        self._index: List[Tuple[int, int]] = []
        for idx, entry in enumerate(self.entries):
            vol = _load_array(entry.volume_path)
            vol = _ensure_4d(vol)
            _, D, H, W = vol.shape
            grid = compute_patch_grid((D, H, W), self.patch_size, self.stride)
            self._grids.append(grid)
            for p_idx in range(grid.num_patches):
                self._index.append((idx, p_idx))

        self._cache_id: Optional[int] = None
        self._cache: Dict[str, np.ndarray] = {}

        self.augment = VesuviusAugmentations(aug_cfg or VesuviusAugConfig()) if augment else None

    def _load_index(self) -> List[IndexEntry]:
        index_path = self.root / "index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing index.csv at {index_path}")

        entries: List[IndexEntry] = []
        with open(index_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                volume_path = (self.root / row["volume_path"]).resolve()
                surface_path = (self.root / row["surface_path"]).resolve() if row.get("surface_path") else None
                ink_path = (self.root / row["ink_path"]).resolve() if row.get("ink_path") else None
                geometry_path = (
                    (self.root / row["geometry_path"]).resolve() if row.get("geometry_path") else None
                )
                spacing = _parse_spacing(row)
                entries.append(
                    IndexEntry(
                        volume_id=row["id"],
                        volume_path=volume_path,
                        surface_path=surface_path,
                        ink_path=ink_path,
                        geometry_path=geometry_path,
                        spacing=spacing,
                        split=row.get("split", "train"),
                    )
                )
        return entries

    def __len__(self) -> int:
        return len(self._index)

    def _load_cached(self, entry_idx: int) -> Dict[str, np.ndarray]:
        if self._cache_id == entry_idx:
            return self._cache

        entry = self.entries[entry_idx]
        vol = _ensure_4d(_load_array(entry.volume_path)).astype(np.float32)
        surface = _load_array(entry.surface_path).astype(np.float32) if entry.surface_path else None
        ink = _load_array(entry.ink_path).astype(np.float32) if entry.ink_path else None
        geom = _load_array(entry.geometry_path).astype(np.float32) if entry.geometry_path else None

        self._cache_id = entry_idx
        self._cache = {"volume": vol, "surface": surface, "ink": ink, "geometry": geom}
        return self._cache

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        entry_idx, patch_idx = self._index[idx]
        entry = self.entries[entry_idx]
        grid = self._grids[entry_idx]
        start = grid.starts[patch_idx].tolist()

        cache = self._load_cached(entry_idx)
        volume = _slice_patch(cache["volume"], start, grid.patch_size)

        sample: Dict[str, Tensor] = {
            "volume": torch.from_numpy(volume).unsqueeze(0),  # (1, C, D, H, W)
        }

        if cache["surface"] is not None:
            surface = _slice_patch(cache["surface"], start, grid.patch_size)
            if surface.ndim == 3:
                surface = surface[None, ...]
            sample["surface_target"] = torch.from_numpy(surface).unsqueeze(0)
            sample["target"] = sample["surface_target"]
        elif cache["ink"] is not None:
            ink = _slice_patch(cache["ink"], start, grid.patch_size)
            if ink.ndim == 3:
                ink = ink[None, ...]
            sample["ink_target"] = torch.from_numpy(ink).unsqueeze(0)
            sample["target"] = sample["ink_target"]
        elif not self.allow_missing_targets:
            raise RuntimeError(f"Missing ink target for {entry.volume_id}")

        if cache["geometry"] is not None:
            geom = _slice_patch(cache["geometry"], start, grid.patch_size)
            if geom.ndim == 3:
                geom = geom[None, ...]
            sample["geometry_target"] = torch.from_numpy(geom).unsqueeze(0)

        if self.augment is not None:
            sample = self.augment(sample)

        # Remove batch dimension before returning.
        out: Dict[str, Tensor] = {
            k: v.squeeze(0).contiguous() for k, v in sample.items()
        }
        out["meta"] = {
            "volume_id": entry.volume_id,
            "patch_start": torch.tensor(start, dtype=torch.long),
        }
        if entry.spacing is not None:
            out["meta"]["spacing"] = torch.tensor(entry.spacing, dtype=torch.float32)
        return out


def _parse_spacing(row: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    """
    Parse voxel spacing from CSV row if present.

    Supported column conventions:
      - spacing_z, spacing_y, spacing_x (preferred)
      - spacing: "z,y,x" or "z y x"
    """
    has_any = any(k in row for k in ("spacing_z", "spacing_y", "spacing_x", "spacing"))
    if not has_any:
        return None

    if row.get("spacing") and not any(row.get(k) for k in ("spacing_z", "spacing_y", "spacing_x")):
        raw = row["spacing"].replace(",", " ").strip()
        parts = [p for p in raw.split() if p]
        if len(parts) == 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        return None

    def _get(name: str) -> float:
        val = row.get(name)
        return float(val) if val not in (None, "") else 1.0

    return (_get("spacing_z"), _get("spacing_y"), _get("spacing_x"))
