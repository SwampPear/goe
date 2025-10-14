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
from urllib.parse import urljoin
from typing import Optional, Dict, Any, List

# progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from src.utils.config import config


def _download_file(session: requests.Session, url: str, scroll_idx: int):
        """Downloads a single file to raw data."""

        # make sure destination exists
        name = url.split("/")[-1]
        dest = Path(config("data", "root"), "raw", str(scroll_idx), name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")

        # position header
        headers = {}
        pos = tmp.stat().st_size if tmp.exists() else 0
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"
        
        # download file
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


class DataServer:
    def __init__(self, scroll: int):
        self.scroll_idx = scroll - 1

        self.base_url = config("data", "urls")[self.scroll_idx]
        self.files = self.list_files()


    def _listdir(self, path: str):
        """Lists directories parsed from autoindex style data server."""
        
        # request page
        url = urljoin(self.base_url, path)
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


    def list_files(self) -> Dict:
        """Lists all files under the most recently update volumes/ dir on the data server."""

        # list files in volumes/
        url = "volumes/"
        paths = self._listdir(url)

        # locate most recent dir
        dated_dirs = []
        for i in range(len(paths)):
            paths[i] = paths[i].rstrip("/").split("/")[-1]

            try:
                dated_dir = int(paths[i])
                dated_dirs.append(dated_dir)
            except Exception:
                pass

        latest_dir = str(max(dated_dirs))

        # list tiff file paths
        url = urljoin(url, latest_dir)

        files = self._listdir(url)

        return {
            "dir": latest_dir,
            "files": files
        }


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
        dest = urljoin(config("data", "root"), "raw", str(self.scroll_idx + 1))

        # concurrent requests
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for file in files:
                out_path = Path(urljoin(dest, file))
                if out_path.exists():
                    continue

                # submit job
                url = self.base_url + "volumes/" + self.files["dir"] + "/" + file
                jobs.append(ex.submit(_download_file, sess, url, self.scroll_idx))

            it = as_completed(jobs)

            # progress bar
            if tqdm is not None:
                it = tqdm(it, total=len(jobs), desc="Downloading", unit="file")
            for fut in it:
                fut.result()



