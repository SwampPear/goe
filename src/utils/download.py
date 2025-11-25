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