from __future__ import annotations
import os
import sys
import time
import math
import hashlib
import requests
from pathlib import Path
import numpy as np
import tifffile as tiff, zarr, numcodecs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any
from urllib.parse import urljoin

try:
    from tqdm import tqdm  # progress bar
except Exception:
    tqdm = None


def _load_tiff_as_zarr(path: str, series: int = 0, level: int = 0, pick_channel: Optional[int] = None,
) -> Tuple[Any, Tuple[float, float], Dict[str, Any]]:
    """
    Returns:
      volume: 3D array shaped (Z, Y, X). Zarr-backed for stacks; memmap-backed for 2D.
      vmin_vmax: inferred intensity range
      meta: {shape, dtype, axes}
    """
    with tiff.TiffFile(path) as tf:
        axes = getattr(tf.series[series], 'axes', None)

        # Fast path: single 2D image (no Z axis) -> memmap + expand to Z=1
        if axes is None or ('Z' not in axes and tf.series[series].shape and len(tf.series[series].shape) == 2):
            mm = tiff.memmap(path)          # (Y, X), zero-copy
            volume = np.expand_dims(mm, 0)      # (1, Y, X)
            meta_axes = 'ZYX'
            dtype = mm.dtype
        else:
            # Zarr-backed for stacks or multi-page
            store = tf.aszarr(series=series, level=level)
            za = zarr.open(store, mode='r')     # zarr.Array
            ax = axes

            # Handle channel by integer indexing (allowed for Zarr)
            if 'C' in ax:
                cpos = ax.index('C')
                if pick_channel is None:
                    if za.shape[cpos] != 1:
                        raise ValueError("Multi-channel TIFF: pass pick_channel")
                    pick_channel = 0
                index = [slice(None)] * za.ndim
                index[cpos] = pick_channel
                za = za[tuple(index)]
                ax = ax.replace('C', '')

            # Ensure we end up (Z, Y, X); use transpose only (no new axes)
            if 'Z' in ax and 'Y' in ax and 'X' in ax:
                order = (ax.index('Z'), ax.index('Y'), ax.index('X'))
                volume = za.transpose(order)
                meta_axes = 'ZYX'
                dtype = za.dtype
            else:
                # Fallback: nonstandard axes -> load via memmap and coerce
                mm = tifffile.memmap(path)
                if mm.ndim == 2:
                    volume = np.expand_dims(mm, 0)
                else:
                    # collapse any leading dims into Z (memmap still ok if contiguous)
                    z = int(np.prod(mm.shape[:-2]))
                    volume = mm.reshape((z, mm.shape[-2], mm.shape[-1]))
                meta_axes = 'ZYX'
                dtype = mm.dtype

    # infer intensity range without touching data
    if np.issubdtype(dtype, np.integer):
        ii = np.iinfo(dtype); vmin_vmax = (float(ii.min), float(ii.max))
    elif np.issubdtype(dtype, np.floating):
        vmin_vmax = (0.0, 1.0)
    else:
        vmin_vmax = (0.0, 1.0)

    meta = {'shape': tuple(volume.shape), 'dtype': str(dtype), 'axes': meta_axes}
    return volume, vmin_vmax, meta

# src/io/loading.py

def _coerce_to_zyx(arr, axes: Optional[str], pick_channel: Optional[int]):
    a = arr
    if axes:
        ax = axes
        if 'C' in ax:
            cpos = ax.index('C')
            if pick_channel is None:
                if a.shape[cpos] != 1:
                    raise ValueError("Multi-channel TIFF: pass pick_channel")
                pick_channel = 0
            # integer index along channel axis is fine for Zarr
            index = [slice(None)] * a.ndim
            index[cpos] = pick_channel
            a = a[tuple(index)]
            ax = ax.replace('C', '')
        if 'Z' in ax:
            order = (ax.index('Z'), ax.index('Y'), ax.index('X'))
            # use transpose (lazy) instead of np.moveaxis
            a = a.transpose(order)
        else:
            # 2D -> add leading Z=1 via reshape (lazy)
            if a.ndim == 2:
                a = a.reshape((1,) + a.shape)     # (1, Y, X)
            else:
                # collapse all but last two into Z; use reshape (lazy)
                newZ = int(np.prod(a.shape[:-2]))
                a = a.reshape((newZ, a.shape[-2], a.shape[-1]))
    else:
        if a.ndim == 2:
            a = a.reshape((1,) + a.shape)          # (1, Y, X)
        else:
            newZ = int(np.prod(a.shape[:-2]))
            a = a.reshape((newZ, a.shape[-2], a.shape[-1]))
    return a
