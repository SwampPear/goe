import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from urllib.parse import urljoin
from src.config import config
from src.io.listing import list_volume_files
from src.io.loading import download_tif


def main():
    # script args
    ap = argparse.ArgumentParser(description="Loads data from the Vesuvius Challenge data server.")
    ap.add_argument("--scroll", type=int, required=True, help="Scroll identifier.")
    ap.add_argument("--dest", required=True, help="Destination for downloaded .tif files.")
    ap.add_argument("--start", type=int, required=True, help="Inclusive start index for download slice.")
    ap.add_argument("--end", type=int, required=True, help="Uninclusive end index for download slice.")
    args = ap.parse_args()

    scroll = args.scroll - 1
    dest = args.dest
    start = args.start
    end = args.end

    # parse autoindex paths from server
    url = config("data", "urls")[scroll]
    paths = list_volume_files(url)

    for i in range(start, end):
        download_tif(paths[i], dest)


if __name__ == '__main__':
    main()
