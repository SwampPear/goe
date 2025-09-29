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

def download_file(session: requests.Session, url: str, out_path: Path, chunk=1<<20):
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


def download_files(paths: List[str], out_dir: str, start: int = 0, count: int = 32, concurrency: int = 4):
    # bound the slice
    end = min(len(paths), start + count)
    block = paths[start:end]

    sess = requests.Session()
    # keep-alive pool
    adapter = requests.adapters.HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency, max_retries=2)
    sess.mount("http://", adapter); sess.mount("https://", adapter)

    jobs = []
    out_dir = Path(out_dir)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for url in block:
            name = url.split("/")[-1]
            out_path = out_dir / name
            if out_path.exists():  # skip existing
                continue
            jobs.append(ex.submit(download_file, sess, url, out_path))

        it = as_completed(jobs)
        if tqdm is not None:
            it = tqdm(it, total=len(jobs), desc="Downloading", unit="file")
        for fut in it:
            fut.result()  # raise on error
            