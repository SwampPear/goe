from __future__ import annotations
import os
import sys
import time
import math
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any
from urllib.parse import urljoin

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def _download_file(session: requests.Session, url: str, out_path: Path, chunk=1<<20):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    # Resume if possible
    headers = {}
    pos = tmp.stat().st_size if tmp.exists() else 0
    if pos > 0:
        headers["Range"] = f"bytes={pos}-"
    with session.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"GET {url} -> {r.status_code}")
        with open(tmp, "ab" if pos > 0 else "wb") as f:
            for chunk_bytes in r.iter_content(chunk_size=chunk):
                if chunk_bytes:
                    f.write(chunk_bytes)
    tmp.replace(out_path)  # atomic move


def download_files(paths: List[str], dest: str, start: int = 0, count: int = 32, concurrency: int = 4):
    # bound path slice
    offset = start + count
    end = min(len(paths), offset)
    paths = paths[start:end]

    # http adapteer (keep-alive pool)
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency, max_retries=2)
    sess.mount("http://", adapter); sess.mount("https://", adapter)

    # submit jobs
    jobs = []
    dest = Path(dest)

    # concurrent requests
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for path in paths:
            # format file path
            name = path.split("/")[-1] # file name
            out_path = dest / name # 
            if out_path.exists():
                continue

            # submit job
            jobs.append(ex.submit(_download_file, sess, path, out_path))

        it = as_completed(jobs)

        if tqdm is not None:
            it = tqdm(it, total=len(jobs), desc="Downloading", unit="file")
        for fut in it:
            fut.result()
            