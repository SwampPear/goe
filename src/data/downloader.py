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


def _safe_url(*parts: str) -> str:
    """
    Join URL parts with single slashes.
    """
    if not parts:
        raise ValueError("No URL parts provided.")
    url = parts[0]
    for part in parts[1:]:
        if not url.endswith("/"):
            url = url + "/"
        url = urljoin(url, str(part).lstrip("/"))
    return url


def _safe_path(*parts: str) -> str:
    """
    Join filesystem path parts safely.
    """
    return str(Path(*parts))


def _listdir(url: str) -> List[str]:
    """
    List directory entries from a simple HTTP index page.
    """
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    entries = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href or href in ("../", "./"):
            continue
        entries.append(href)
    return entries


def _download_file(sess: requests.Session, out_path: Path, url: str) -> None:
    """
    Download a single file to disk with a temporary .part file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists():
        return
    with sess.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp_path.replace(out_path)


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


class VesuviusChallengeSegmentDatasetDownloader:
    def __init__(self, scroll: int):
        pass

    def list_files(self) -> Dict[str, Any]:
        pass

    def download_files(self, start: int = 0, count: int = 1, concurrency: int = 4):
        pass
