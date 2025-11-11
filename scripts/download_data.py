import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from urllib.parse import urljoin
from src.utils.config import config
from src.data.dataset import VesuviusChallengeVolumeDatasetDownloader

def main():
    ap = argparse.ArgumentParser(description="Downloads data from the Vesuvius Challenge data server.")
    ap.add_argument("--scroll", type=int, required=True, help="Scroll identifier.")
    ap.add_argument("--start", type=int, required=True, help="Start index.")
    ap.add_argument("--count", type=int, required=True, help="File count.")
    args = ap.parse_args()

    scroll = args.scroll
    start = args.start
    count = args.count

    downloader = VesuviusChallengeVolumeDatasetDownloader(scroll)
    downloader.download_files(start=start, count=count)

if __name__ == '__main__':
    main()
