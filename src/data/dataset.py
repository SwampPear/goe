from __future__ import annotations
import os
import sys
import time
import math
import hashlib
import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
import tifffile as tiff, zarr, numcodecs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from src.utils.config import config


def _safe_path(*args) -> str:
    """
    Safely formats a path as a string.
    Args:
        args: str | Path
    Returns:
        formatted path
    """
    if len(args) < 1:
        raise Exception("'_safe_path' must be given at least 1 argument.")

    out = Path(args[0])
    for arg in args[1:]:
        if isinstance(arg, str):
            out = out / Path(arg)
        elif isinstance(arg, Path):
            out = out / arg
        else:
            raise Exception("'_safe_path' must be given str or Path arguments.")

    return str(out)


def _safe_url(*args) -> str:
    """
    Safely formats a url as a string.
    Args:
        args: str | Path
    Returns:
        formatted url
    """
    if len(args) < 1:
        raise Exception("'_safe_url' must be given at least 1 argument.")

    out = _safe_path(args)
    out = out.replace("https:/", "https://").replace("https:/", "https://")

    return out


def _download_file(session: requests.Session, url: str, scroll_idx: int):
        """Downloads a single file to raw data."""

        # make sure destination exists
        dest = Path(config("data", "root"), "raw", str(scroll_idx), url.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        # position header
        headers = {}
        pos = tmp.stat().st_size if tmp.exists() else 0
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"
        
        # download file
        url = str(url).replace("https:/", "https://")
        with session.get(url, stream=True, timeout=60, headers=headers) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(f"GET {url} -> {r.status_code}")
            
            # write chunk
            with open(tmp, "ab" if pos > 0 else "wb") as f:
                for chunk_bytes in r.iter_content(chunk_size=1<<20):
                    if chunk_bytes:
                        f.write(chunk_bytes)

        # atomic move
        tmp.replace(dest)


def _listdir(url: str):
        """Lists directories parsed from autoindex style data server."""
        
        # request page
        res = requests.get(url)
        res.raise_for_status()
        html = res.text

        # parse paths
        soup = BeautifulSoup(html, "html.parser")
        paths = []

        for tr in soup.select("#list tbody tr"):
            a = tr.select_one("td.link a")

            if not a:
                continue

            name = a.get_text(strip=True)
            paths.append(name)
            
        return paths

class VesuviusChallengeDatasetDownloader:
    def __init__(self, scroll: int):
        self.scroll_idx = scroll - 1
        self.base_url = Path(config("data", "urls")[self.scroll_idx])
        self.files = self.list_files()


    def list_files(self) -> Dict:
        """Lists all files under the most recently update volumes/ dir on the data server."""

        # list files in volumes/
        url = self.base_url / Path("volumes")
        url = str(url).replace("https:/", "https://")
        paths = _listdir(url)

        # locate most recent dir
        dated_dirs = []
        for i in range(len(paths)):
            paths[i] = paths[i].rstrip("/").split("/")[-1]

            # get date from path
            try:
                dated_dir = int(paths[i])
                dated_dirs.append(dated_dir)
            except Exception:
                pass

        latest_dir = Path(str(max(dated_dirs)))

        # list tiff file paths
        url = url / latest_dir
        url = str(url).replace("https:/", "https://")

        files = _listdir(url)

        return {"dir": latest_dir, "files": files}


    def download_files(self, start: int = 0, count: int = 4, concurrency: int = 4):
        """Concurrently downloads files from the data server."""
        # path slice
        end = min(len(self.files["files"]), start + count)
        files = self.files["files"][start:end]

        # http adapteer (keep-alive pool)
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=concurrency, pool_maxsize=concurrency, max_retries=2)
        sess.mount("http://", adapter); sess.mount("https://", adapter)

        # submit jobs
        jobs = []
        dest = Path(config("data", "root")) / Path("raw") / Path("scan") / Path(str(self.scroll_idx + 1))

        # concurrent requests
        file_dir = self.base_url / Path("volumes") / self.files["dir"]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for file in files:
                out_path = dest / Path(file)
                if out_path.exists():
                    continue

                # submit job
                url = file_dir / Path(file)
                jobs.append(ex.submit(_download_file, sess, url, self.scroll_idx))

            it = as_completed(jobs)

            # progress bar
            it = tqdm(it, total=len(jobs), desc="Downloading", unit="file")
            for fut in it:
                fut.result()