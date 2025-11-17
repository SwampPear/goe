"""
Dataset downloaders for the Vesuvius Challenge data server.

This module provides safe path/URL helpers, HTML directory parsing, and concurrent HTTP downloading for the Vesuvius 
Challenge datasets (volumes, fragments, segments). Directory listings are parsed from typical autoindex-style HTTP 
directory pages (e.g., Nginx autoindex).

Downloaders:
    - VesuviusChallengeVolumeDatasetDownloader:
        Finds the newest `volumes/<date>/` directory and downloads its constituent files.
    - VesuviusChallengeFragmentDatasetDownloader:
         Placeholder for fragment-level datasets.
    - VesuviusChallengeSegmentDatasetDownloader:
         Placeholder for segment-level datasets.

Configuration is sourced via `src.utils.config.config`, which must
provide:
    config("data", "root")        -> base download directory
    config("data", "scroll_urls") -> mapping of scroll_id → base URL

This module is designed for use in offline dataset preparation and training pipelines that require local copies of 
the Vesuvius Challenge data.
"""

from __future__ import annotations
import os, sys, time, math, hashlib
import requests
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from src.utils.config import config


def _safe_path(*parts: str | Path) -> str:
    """
    Join path file path parts together.
    Args:
        parts: *(str | Path) - argument list for path parts
    Returns:
        the formatted path
    """
    if not parts:
        raise ValueError("'_safe_path' needs at least one argument")
    p = Path()
    for part in parts:
        if isinstance(part, (str, Path)):
            p = p / Path(part)
        else:
            raise TypeError("path parts must be str or Path")
    return str(p)


def _safe_url(*parts: str) -> str:
    """
    Join URL parts without mangling the scheme or inserting per-character slashes.
    Args:
        parts: *str - argument list for url parts
    Returns:
        the formatted url
    """
    if not parts:
        raise ValueError("'_safe_url' needs at least one argument")
    base = parts[0]
    # ensure base ends with '/', so urljoin appends instead of replacing
    if not base.endswith('/'):
        base += '/'
    out = base
    for seg in parts[1:]:
        seg = seg.lstrip('/')  # avoid resetting path
        out = urljoin(out, seg + ('/' if seg and not seg.endswith('/') else ''))
    # drop the trailing slash we added unless caller intended it
    return out[:-1] if out.endswith('/') else out


def _ensure_dest_exists(dest: Path) -> tuple[Path, Path]:
    """
    Ensures a destination file exists and creates it if not.
    Args:
        dest: Path - the destination file
    Returns:
        destination file, temporary destination (for download)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    return dest, tmp


def _download_file(session: requests.Session, dest: str, url: str):
    """
    Downloads a single file.
    Args:
        session: requests.Session - http session
        dest: str - destination file path
        url: str - download url
    """
    dest, tmp = _ensure_dest_exists(dest)

    headers = {}
    pos = tmp.stat().st_size if tmp.exists() else 0
    if pos > 0:
        headers["Range"] = f"bytes={pos}-"

    with session.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"GET {url} -> {r.status_code}")
        with open(tmp, "ab" if pos > 0 else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)

    tmp.replace(dest)


def _download_files(files: List[str], dest_dir: str, base_url: str, start: int, count: int, concurrency: int):
    """
    Downloads a sequence of files.
    Args:
        files: str[] - file paths to download
        dest_dir: str - destination directory
        base_url: str - base url of download
        start: int - start index of files
        count: int - number of files to download
        concurrency: int - number of concurrent workers to use
    """
    end = min(len(files), start + count)
    files = files[start:end]

    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=concurrency, pool_maxsize=concurrency, max_retries=2
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    jobs = []
    dest_dir_p = Path(dest_dir)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for fname in files:
            out_path = dest_dir_p / fname
            if out_path.exists():
                continue
            file_url = _safe_url(base_url, fname)
            
            jobs.append(ex.submit(_download_file, sess, out_path, file_url))

        it = tqdm(as_completed(jobs), total=len(jobs), desc="Downloading", unit="file")
        for fut in it:
            fut.result()


def _listdir(url: str) -> List[str]:
    """
    Lists all files from an autoindex style web directory.
    Args:
        url: str - url to parse
    Returns:
        list of files
    """
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    paths = []
    # Nginx autoindex variants: try common selectors, then fallback
    anchors = soup.select("#list tbody tr td.link a") or soup.select("pre a") or soup.select("a")
    for a in anchors:
        name = a.get_text(strip=True)
        if not name or name in (".", ".."):
            continue
        paths.append(name)
    return paths


def _max_date_dir(paths: List[str]) -> str:
    """
    Gets the max dated directory from a list of dated directories.
    Args:
        paths: str[] - file paths
    Returns:
        max dated directory
    """
    candidates = []
    for s in paths:
        s = s.rstrip("/")
        try:
            candidates.append(int(s))
        except ValueError:
            pass
    if not candidates:
        raise RuntimeError("No dated directories found")
    return str(max(candidates))


class VesuviusChallengeVolumeDatasetDownloader:
    def __init__(self, scroll: int):
        self.scroll = scroll
        self.base_url = _safe_url(config("data", "scroll_urls")[self.scroll])
        self.files = self.list_files()

    def list_files(self) -> Dict[str, Any]:
        """
        List files under the most recent 'volumes/' dir.
        Returns:
            {
                'dir': file directory
                'files': file paths
            }
        """
        volumes_url = _safe_url(self.base_url, "volumes")
        date_dirs = _listdir(volumes_url)
        latest_dir = _max_date_dir(date_dirs)

        latest_url = _safe_url(volumes_url, latest_dir)
        files = _listdir(latest_url)

        files = [f.rstrip("/") for f in files if f and not f.endswith("/")]
        return {"dir": latest_dir, "files": files}


    def download_files(self, start: int = 0, count: int = 1, concurrency: int = 4):
        """
        Downloads files from data server's 'volumes/' dir.
        Args:
            start: int - start index of files
            count: int - number of files to download
            concurrency: int - number of concurrent workers to use
        """
        dest = _safe_path(config("data", "root"), "raw", "volumes", str(self.scroll))
        base = _safe_url(self.base_url, "volumes", self.files["dir"])
        _download_files(self.files["files"], dest, base, start, count, concurrency)


class VesuviusChallengeFragmentDatasetDownloader:
    def __init__(self, scroll: int):
        pass

    def list_files(self) -> Dict[str, Any]:
        pass

    def download_files(self, start: int = 0, count: int = 1, concurrency: int = 4):
        pass


class VesuviusChallengeSegmentDatasetDownloader:
    def __init__(self, scroll: int):
        pass

    def list_files(self) -> Dict[str, Any]:
        pass

    def download_files(self, start: int = 0, count: int = 1, concurrency: int = 4):
        pass
