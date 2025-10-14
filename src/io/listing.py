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

from src.config import config


class DataConnection:
    def __init__(self, scroll):
        self.scroll_idx = scroll - 1

        self.base_url = config("data", "urls")[scroll]
        self.files = self.list_files()

        print(self.files)

    def _listdir(self, path):
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


    def list_files(self) -> List[str]:
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

        latest_dir = max(dated_dirs)

        # list tiff file paths
        url = urljoin(url, str(latest_dir))
        files = self._listdir(url)

        return files



