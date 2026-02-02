from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


SCROLL_URLS = [
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/",
    "https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/",
    "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/",
    "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/",
    "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/",
]

DEFAULT_DATA_ROOT = Path("data")


def _safe_url(*parts: str) -> str:
    """Join URL parts with single slashes."""
    if not parts:
        raise ValueError("No URL parts provided.")
    url = parts[0]
    for part in parts[1:]:
        if not url.endswith("/"):
            url = url + "/"
        url = urljoin(url, str(part).lstrip("/"))
    return url


def _listdir(url: str) -> List[str]:
    """List directory entries from a simple HTTP index page."""
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


def _download_file(
    sess: requests.Session,
    out_path: Path,
    url: str,
    max_retries: int = 3,
) -> None:
    """Download a single file with byte-range resume and retry on failure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            existing_bytes = 0
            if tmp_path.exists():
                existing_bytes = tmp_path.stat().st_size
                headers["Range"] = f"bytes={existing_bytes}-"

            with sess.get(url, stream=True, headers=headers, timeout=60) as resp:
                if resp.status_code == 416:
                    # Range not satisfiable — file already complete
                    tmp_path.replace(out_path)
                    return
                resp.raise_for_status()

                mode = "ab" if resp.status_code == 206 else "wb"
                with open(tmp_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            tmp_path.replace(out_path)
            return

        except (requests.RequestException, OSError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning(
                    "Retry %d/%d for %s (%s), waiting %ds",
                    attempt, max_retries, out_path.name, exc, wait,
                )
                time.sleep(wait)
            else:
                raise


def _download_files(
    files: List[str],
    dest_dir: Path,
    base_url: str,
    start: int,
    count: int,
    concurrency: int,
) -> None:
    """Download a slice of files concurrently with progress tracking."""
    end = min(len(files), start + count)
    files = files[start:end]

    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=concurrency, pool_maxsize=concurrency,
    )
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)

    # Track progress so restarts can report what's left
    manifest = _load_manifest(dest_dir)
    skipped = 0
    jobs = []

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for fname in files:
            out_path = dest_dir / fname
            if out_path.exists():
                skipped += 1
                manifest[fname] = "done"
                continue
            file_url = _safe_url(base_url, fname)
            jobs.append((fname, ex.submit(_download_file, sess, out_path, file_url)))

        if skipped:
            log.info("Skipped %d already-downloaded files", skipped)

        done = 0
        failed = []
        for fname, fut in jobs:
            try:
                fut.result()
                manifest[fname] = "done"
                done += 1
            except Exception as exc:
                manifest[fname] = f"failed: {exc}"
                failed.append(fname)
                log.error("Failed to download %s: %s", fname, exc)
            finally:
                _save_manifest(dest_dir, manifest)

        total = skipped + done + len(failed)
        log.info(
            "Download complete: %d/%d succeeded, %d skipped, %d failed",
            done, total, skipped, len(failed),
        )
        if failed:
            raise RuntimeError(
                f"{len(failed)} files failed to download. "
                f"Re-run to retry. Failed: {failed[:10]}"
            )


MANIFEST_NAME = ".download_manifest.json"


def _load_manifest(dest_dir: Path) -> Dict[str, str]:
    path = dest_dir / MANIFEST_NAME
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(dest_dir: Path, manifest: Dict[str, str]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / MANIFEST_NAME
    with open(path, "w") as f:
        json.dump(manifest, f)


def _max_date_dir(paths: List[str]) -> str:
    """Return the highest-numbered (most recent) dated directory name."""
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


class VesuviusVolumeDownloader:
    """Download volume slices for a given scroll from dl.ash2txt.org."""

    def __init__(
        self,
        scroll: int,
        data_root: Optional[Path] = None,
        scroll_urls: Optional[List[str]] = None,
    ):
        self.scroll = scroll
        urls = scroll_urls or SCROLL_URLS
        if scroll < 0 or scroll >= len(urls):
            raise ValueError(f"scroll must be in 0..{len(urls) - 1}, got {scroll}")
        self.base_url = _safe_url(urls[scroll])
        self.data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT
        self.files = self._list_files()

    def _list_files(self) -> Dict[str, Any]:
        """List files under the most recent volumes/ directory."""
        volumes_url = _safe_url(self.base_url, "volumes")
        date_dirs = _listdir(volumes_url)
        latest_dir = _max_date_dir(date_dirs)

        latest_url = _safe_url(volumes_url, latest_dir)
        files = _listdir(latest_url)
        files = [f.rstrip("/") for f in files if f and not f.endswith("/")]
        return {"dir": latest_dir, "files": files}

    def download(
        self,
        start: int = 0,
        count: int = 1,
        concurrency: int = 4,
    ) -> None:
        """Download volume files.

        Args:
            start: Index of first file to download.
            count: Number of files to download.
            concurrency: Number of concurrent download workers.
        """
        dest = self.data_root / "raw" / "volumes" / str(self.scroll)
        base = _safe_url(self.base_url, "volumes", self.files["dir"])
        _download_files(self.files["files"], dest, base, start, count, concurrency)
