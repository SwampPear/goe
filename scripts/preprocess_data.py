import argparse
import csv
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _find_files(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    for ext in exts:
        yield from root.rglob(f"*{ext}")


def _load_volume(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            key = "arr_0" if "arr_0" in z else list(z.keys())[0]
            return z[key]
    if path.suffix in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    if path.suffix == ".zarr" or path.is_dir():
        import zarr
        return np.asarray(zarr.open(path, mode="r"))
    raise ValueError(f"Unsupported input format: {path}")


def _ensure_4d(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 3:
        return volume[None, ...]
    if volume.ndim == 4:
        return volume
    raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")


def _maybe_normalize(volume: np.ndarray, mode: Optional[str]) -> np.ndarray:
    if mode is None:
        return volume
    if mode == "minmax":
        vmin = float(np.min(volume))
        vmax = float(np.max(volume))
        denom = max(vmax - vmin, 1e-6)
        return (volume - vmin) / denom
    if mode == "zscore":
        mean = float(np.mean(volume))
        std = float(np.std(volume))
        return (volume - mean) / max(std, 1e-6)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _match_label(stem: str, label_dir: Optional[Path]) -> Optional[Path]:
    if label_dir is None:
        return None
    for ext in (".npy", ".npz", ".tif", ".tiff"):
        cand = label_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess Vesuvius volumes into a patch-ready dataset.")
    ap.add_argument("--raw-root", type=str, default="data/raw/volumes", help="Raw download root.")
    ap.add_argument("--out-root", type=str, default="data/processed", help="Processed output root.")
    ap.add_argument("--ink-dir", type=str, default=None, help="Optional ink label directory.")
    ap.add_argument("--geometry-dir", type=str, default=None, help="Optional geometry label directory.")
    ap.add_argument("--val-split", type=float, default=0.1, help="Fraction of volumes for validation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", type=str, default=None, help="Optional: minmax or zscore")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_vol_dir = out_root / "volumes"
    out_vol_dir.mkdir(parents=True, exist_ok=True)

    ink_dir = Path(args.ink_dir) if args.ink_dir else None
    geom_dir = Path(args.geometry_dir) if args.geometry_dir else None

    files = list(_find_files(raw_root, (".npy", ".npz", ".tif", ".tiff", ".zarr")))
    if not files:
        raise RuntimeError(f"No raw volumes found in {raw_root}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    val_count = int(len(files) * args.val_split)
    val_set = set(f.name for f in files[:val_count])

    index_rows = []
    for src in files:
        stem = src.stem.replace("__volume", "")
        volume = _load_volume(src)
        volume = _ensure_4d(volume).astype(np.float32)
        volume = _maybe_normalize(volume, args.normalize)

        out_path = out_vol_dir / f"{stem}.npy"
        np.save(out_path, volume)

        ink_path = _match_label(stem, ink_dir)
        geom_path = _match_label(stem, geom_dir)

        split = "val" if src.name in val_set else "train"
        index_rows.append(
            {
                "id": stem,
                "split": split,
                "volume_path": str(out_path.relative_to(out_root)),
                "ink_path": str(ink_path.relative_to(out_root)) if ink_path else "",
                "geometry_path": str(geom_path.relative_to(out_root)) if geom_path else "",
            }
        )

    index_path = out_root / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "split", "volume_path", "ink_path", "geometry_path"]
        )
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"Wrote {len(index_rows)} entries to {index_path}")


if __name__ == "__main__":
    main()
