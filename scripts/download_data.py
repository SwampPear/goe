import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from urllib.parse import urljoin
from src.config import config
from src.io.listing import list_volume_files
from src.io.loading import download_files


def main():
    # args
    ap = argparse.ArgumentParser(description="Loads data from the Vesuvius Challenge data server.")
    ap.add_argument("--scroll", type=int, required=True, help="Scroll identifier.")
    ap.add_argument("--dest", required=True, help="Destination for downloaded .tif files.")
    ap.add_argument("--start", type=int, required=True, help="Start index for download slice.")
    ap.add_argument("--count", type=int, required=True, help="Count of files to download in download slice.")
    args = ap.parse_args()

    scroll = args.scroll - 1
    dest = args.dest
    start = args.start
    count = args.count

    url = config("data", "urls")[scroll]

    paths = list_volume_files(url)
    download_files(paths, dest, scroll, start=start, count=count)


if __name__ == '__main__':
    main()
