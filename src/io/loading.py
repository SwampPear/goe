from __future__ import annotations
import os
import sys
import time
import math
import hashlib
import requests
from typing import Optional, Dict, Any
from urllib.parse import urljoin

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class DownloadError(RuntimeError):
    pass


def _fmt_bytes(n: int) -> str:
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units)-1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {units[i]}"


def download_tif(
    url: str,
    dest_path: str,
    *,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.7,
    chunk_size: int = 1 << 20,  # 1 MiB
    expected_sha256: Optional[str] = None,
    auth: Optional[Any] = None,     # e.g. ('user','pass') or requests.AuthBase
    cookies: Optional[Dict[str, str]] = None,
    verify_tls: bool = True,
) -> str:
    """
    Download a .tif from an HTTP(S) data server with resume & integrity checks.

    Args:
      base_url: e.g. "https://data.yourserver.org/"
      remote_path: e.g. "volumes/scroll2/000123.tif" (or "/volumes/...")
      dest_path: local file to write to (created if needed)
      timeout: per-request timeout (seconds)
      retries: max retry attempts on transient failures
      backoff: exponential backoff factor between retries
      chunk_size: streaming read chunk size in bytes
      expected_sha256: optional hex digest to verify after download
      headers, auth, cookies: passed through to requests
      verify_tls: True to verify HTTPS certificates

    Returns:
      dest_path (string)

    Raises:
      DownloadError on failure or checksum mismatch.
    """

    # url
    if not url.lower().endswith((".tif", ".tiff")):
        raise DownloadError(f"URL does not look like a TIFF: {remote_path}")

    # file name
    fname = url.split("/")[-1]
    dest_path += fname

    # ensure destination directory
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

    # session checks if server supports head
    sess = requests.Session()
    headers = {"User-Agent": "scroll-rmoe/1.0"}

    # best effort HEAD request
    content_length = None
    accept_ranges = False
    content_type = None

    try:
        hr = sess.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        if hr.ok:
            content_length = int(hr.headers.get("Content-Length")) if hr.headers.get("Content-Length") else None
            accept_ranges = hr.headers.get("Accept-Ranges", "").lower() == "bytes"
            content_type = hr.headers.get("Content-Type", "")
    except requests.RequestException:
        pass  # Not fatal; we can proceed without HEAD

    if content_type and "tif" not in content_type.lower():
        sys.stderr.write(f"[warn] Content-Type is '{content_type}', expected TIFF.\n")

    # Resume if possible
    mode = "wb"
    resume_from = 0
    if os.path.exists(dest_path) and accept_ranges:
        resume_from = os.path.getsize(dest_path)
        if content_length is not None and resume_from >= content_length:
            # Already complete; verify checksum if provided
            if expected_sha256:
                _verify_sha256(dest_path, expected_sha256)
            return dest_path
        if resume_from > 0:
            mode = "ab"

    # Retry loop
    attempt = 0
    while True:
        attempt += 1
        try:
            # Range header for resume
            h = dict(headers)
            if resume_from > 0:
                h["Range"] = f"bytes={resume_from}-"

            with sess.get(
                url, stream=True, timeout=timeout, headers=h,
                auth=auth, cookies=cookies, verify=verify_tls, allow_redirects=True
            ) as r:
                if r.status_code in (200, 206):
                    total_bytes = content_length
                    # If partial content, server may give total via Content-Range
                    if r.status_code == 206:
                        cr = r.headers.get("Content-Range", "")
                        # Format: bytes START-END/TOTAL
                        if "/" in cr:
                            try:
                                total_bytes = int(cr.split("/")[-1])
                            except Exception:
                                pass

                    # Progress setup
                    file_size_known = total_bytes is not None
                    desc = f"Downloading {os.path.basename(dest_path)}"
                    pbar = None
                    if tqdm is not None:
                        pbar = tqdm(
                            total=None if total_bytes is None else total_bytes,
                            initial=resume_from,
                            unit="B", unit_scale=True, unit_divisor=1024,
                            desc=desc, ascii=True,
                        )

                    sha = hashlib.sha256() if expected_sha256 else None
                    written = resume_from

                    with open(dest_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if not chunk:
                                continue
                            f.write(chunk)
                            written += len(chunk)
                            if sha:
                                sha.update(chunk)
                            if pbar:
                                pbar.update(len(chunk))

                    if pbar:
                        pbar.close()

                    # Size sanity check (if known)
                    if file_size_known and written != total_bytes:
                        raise DownloadError(f"Incomplete download: expected {_fmt_bytes(total_bytes)}, got {_fmt_bytes(written)}")

                    # Checksum
                    if expected_sha256:
                        _verify_sha256(dest_path, expected_sha256)

                    return dest_path

                else:
                    raise DownloadError(f"HTTP {r.status_code}: {r.reason}")

        except (requests.RequestException, DownloadError) as e:
            if attempt >= retries:
                raise DownloadError(f"Failed to download after {retries} attempts: {e}")
            sleep_for = backoff ** (attempt - 1)
            sys.stderr.write(f"[retry {attempt}/{retries}] {e} — retrying in {sleep_for:.1f}s\n")
            time.sleep(sleep_for)


def _verify_sha256(path: str, expected_hex: str) -> None:
    expected_hex = expected_hex.lower().strip()
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    got = sha.hexdigest()
    if got != expected_hex:
        raise DownloadError(f"SHA256 mismatch for {path}: expected {expected_hex}, got {got}")