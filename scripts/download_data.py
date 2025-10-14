import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import argparse
from urllib.parse import urljoin
from src.config import config
from src.io.listing import DataConnection
from src.io.loading import download_files


def main():
    # args
    ap = argparse.ArgumentParser(description="Loads data from the Vesuvius Challenge data server.")
    ap.add_argument("--scroll", type=int, required=True, help="Scroll identifier.")
    ap.add_argument("--start", type=int, required=True, help="Start index for download slice.")
    ap.add_argument("--count", type=int, required=True, help="Count of files to download in download slice.")
    args = ap.parse_args()

    scroll = args.scroll
    start = args.start
    count = args.count

    DataConnection(scroll)


if __name__ == '__main__':
    main()
