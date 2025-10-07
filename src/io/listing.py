from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
from typing import List


def _autoindex_listdir(url: str) -> List[str]:
    """
    List dirs from a parsed autoindex style website.

    Args:
        url - website url
    """
    # request html page
    res = requests.get(url, headers={"User-Agent": "scroll-rmoe/1.0"})
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

def list_volume_files(url: str) -> List[str]:
    """
    Lists all files under the latest volume dir on a the Vesuvius data server.

    Args:
        url - website url
    """

    # list files in volumes/
    url = urljoin(url.rstrip("/") + "/", "volumes/")
    paths = _autoindex_listdir(url)

    # format timestamps and select latest acquisition
    for i in range(len(paths)):
        paths[i] = paths[i].rstrip("/").split("/")[-1]

        if paths[i] == ".vckeep":
            paths.remove(paths[i])

    latest = max(paths, key=lambda p: int(p))

    # update url to acquisition directory and acquire file paths
    url = urljoin(url, str(latest) + "/")
    paths = _autoindex_listdir(url)

    # format full path names
    for i in range(len(paths)):
        paths[i] = url + paths[i]

    return paths