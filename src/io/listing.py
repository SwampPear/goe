from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from typing import Optional, List, Dict


def autoindex_parse_paths(html: str):
    """Parses autoindex paths from a data server page."""
    soup = BeautifulSoup(html, "html.parser")

    paths = []

    for tr in soup.select("#list tbody tr"):
        a = tr.select_one("td.link a")

        if not a:
            continue

        name = a.get_text(strip=True)

        paths.append(name)
        
    return paths


def autoindex_listdir(
    base_url: str,
    remote_dir: str,
    *,
    exts = (".tif", ".tiff"),
    timeout: float = 20.0,
) -> List[Dict]:
    """
    List files from a remote HTTP directory (autoindex-style).

    Returns [{'name': '000123.tif', 'url': '...', 'size': None, 'modified': None}, ...]
    """

    # request html page
    url = urljoin(base_url.rstrip("/") + "/", remote_dir.lstrip("/") + "/")
    headers = {"User-Agent": "scroll-rmoe/1.0"}
    res = requests.get(url, timeout=timeout, headers=headers)
    res.raise_for_status()
    html = res.text

    # parse paths
    paths = autoindex_parse_paths(html)

    print(paths)

    """
    # Try JSON first (some endpoints return JSON)
    try:
        js = r.json()
        items = []
        for it in js:
            name = it.get("name") or it.get("Key") or it.get("filename")
            if not name: continue
            if name.lower().endswith(exts):
                items.append({
                    "name": name,
                    "url": urljoin(url, name),
                    "size": it.get("size") or it.get("Size"),
                    "modified": it.get("last_modified") or it.get("LastModified"),
                })
        if items:
            return sorted(items, key=lambda d: d["name"])
    except Exception:
        pass

    # Fallback: parse HTML links (no BeautifulSoup needed)
    import re
    hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', text, flags=re.I)
    files = []
    for href in hrefs:
        # ignore parent links and anchors
        if href in ("../", "./") or href.startswith("#"):
            continue
        name = href.split("?")[0].split("#")[0]
        # Only direct filenames in this directory; skip subdirs unless you want them
        if name.endswith("/") or "/" in name:
            continue
        if name.lower().endswith(exts):
            files.append({"name": name, "url": urljoin(url, name), "size": None, "modified": None})
    # Natural sort by file name with numbers
    def nkey(d):
        s = d["name"]
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]
    return sorted(files, key=nkey)
    """