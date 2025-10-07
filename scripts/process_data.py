import sys
from pathlib import Path

# add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List
from src.config import config
from src.io.loading import download_files, load_tiff_as_zarr
from src.io.saving import write_tfrecord
from src.processing.features import process_features


def process_tiff_batch(files: List[Path], scroll_id: int) -> None:
    """
    Iterate and process each TIFF. Replace the 'process' stub with your pipeline.
    """
    for fp in files:
        try:
            vol, intensity_range, meta = load_tiff_as_zarr(fp)

            features = process_features(volume=vol, intensity_range=intensity_range)

            write_tfrecord(
                features,
                path=f"data/processed/{scroll_id}/{fp.split('/')[-1].replace('.tif', '')}.tfrecord",
                scroll_id=scroll_id,
                intensity_range=intensity_range
            )

            print(f"[OK] Processed: {fp} | shape={getattr(vol, 'shape', None)} dtype={getattr(vol, 'dtype', None)}")
        except Exception as e:
            raise e
            print(f"[ERR] Failed processing {fp}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Download & process a range of TIFFs from the Vesuvius data server.")
    ap.add_argument("--scroll", type=int, required=True, help="Scroll identifier.")
    ap.add_argument("--start", type=int, required=True, help="Start index (0-based) within the remote listing.")
    ap.add_argument("--count", type=int, required=True, help="Number of files to handle from the start index.")
    ap.add_argument("--dest", type=str, default="data/raw", help="Destination root directory.")
    args = ap.parse_args()

    scroll = args.scroll
    start = args.start
    count = args.count
    dest_root = args.dest

    dest = dest_root + f"/{scroll - 1}"

    file_paths = [dest + "/" + ("0" * (5 - len(str(i)))) + str(i) + ".tif" for i in range(start, start + count)]

    process_tiff_batch(file_paths, scroll - 1)


if __name__ == "__main__":
    main()
