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
from tqdm import tqdm


def load_tiff_as_zarr(
    path: str,
    series: int = 0,
    level: int = 0,
    pick_channel: Optional[int] = None,
) -> Tuple[Any, Tuple[float, float], Dict[str, Any]]:
    """
    Simplified loader for common TIFF layouts.
    Returns:
      volume: (Z, Y, X) array (zarr-backed for stacks, memmap-backed for 2D)
      vmin_vmax: inferred intensity range
      meta: {'shape', 'dtype', 'axes'}
    """
    with tiff.TiffFile(path) as tf:
        s = tf.series[series]
        axes = getattr(s, 'axes', None)

        # zarr-backed stack
        za = zarr.open(tf.aszarr(series=series, level=level), mode='r')
        ax = axes

        # collect Z, Y, X in that order
        wanted = ['Z', 'Y', 'X']
        if not all(a in ax for a in wanted):
            raise ValueError(f"Unsupported axes layout: {axes!r} (need Z, Y, X present)")
        order = tuple(ax.index(a) for a in wanted)
        volume = za.transpose(order)
        dtype = za.dtype
        meta_axes = 'ZYX'
            
    # intensity range
    if np.issubdtype(dtype, np.integer):
        ii = np.iinfo(dtype); vmin_vmax = (float(ii.min), float(ii.max))
    else:
        vmin_vmax = (0.0, 1.0)

    meta = {'shape': tuple(volume.shape), 'dtype': str(dtype), 'axes': meta_axes}
    
    return volume, vmin_vmax, meta
