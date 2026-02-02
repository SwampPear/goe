from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import tifffile

from scripts.common import build_model, set_seed
from src.utils.segmentation import binarize_logits, resize_logits


def _parse_tuple3(arg: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected format D,H,W (comma-separated).")
    return parts[0], parts[1], parts[2]


def _load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        with np.load(path) as z:
            key = "arr_0" if "arr_0" in z else list(z.keys())[0]
            return z[key]
    if path.suffix in (".tif", ".tiff"):
        return tifffile.imread(path)
    raise ValueError(f"Unsupported array format: {path}")


def _ensure_4d(volume: np.ndarray) -> np.ndarray:
    if volume.ndim == 3:
        return volume[None, ...]
    if volume.ndim == 4:
        return volume
    raise ValueError(f"Expected 3D or 4D volume, got shape {volume.shape}")


def _load_index(root: Path, split: str) -> List[dict]:
    index_path = root / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.csv at {index_path}")
    rows: List[dict] = []
    with open(index_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split and row.get("split", "train") != split:
                continue
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict and build submission.zip.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--limit-volumes", type=int, default=None)
    parser.add_argument("--stem-patch-size", type=_parse_tuple3, default="4,4,4")
    parser.add_argument("--stem-stride", type=_parse_tuple3, default="2,2,2")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny config for quick tests.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dtype", type=str, default="uint8")
    parser.add_argument("--match-input-dtype", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("submission.zip"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    model = build_model(
        stem_patch_size=args.stem_patch_size,
        stem_stride=args.stem_stride,
        tiny=args.tiny,
    ).to(device)

    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    rows = _load_index(args.data_root, args.split)
    if args.limit_volumes is not None:
        rows = rows[: int(args.limit_volumes)]

    if not rows:
        raise RuntimeError(f"No rows found for split='{args.split}'.")

    model.eval()
    tmp_dir = args.out.with_suffix("")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for row in rows:
            vol_path = (args.data_root / row["volume_path"]).resolve()
            volume = _ensure_4d(_load_array(vol_path)).astype(np.float32)
            vol_id = row["id"]
            _, D, H, W = volume.shape

            x = torch.from_numpy(volume).unsqueeze(0).to(device)  # [1, C, D, H, W]
            out = model(x)
            logits = out["logits"]
            if logits.dim() != 5:
                raise RuntimeError(f"Expected 5D logits, got {tuple(logits.shape)}")

            logits_up = resize_logits(logits, (D, H, W), mode="trilinear")
            mask = binarize_logits(logits_up, threshold=args.threshold)[0, 0]

            if args.match_input_dtype:
                out_dtype = volume.dtype
            else:
                out_dtype = np.dtype(args.out_dtype)

            mask_np = mask.cpu().numpy().astype(out_dtype)
            out_path = tmp_dir / f"{vol_id}.tif"
            tifffile.imwrite(out_path, mask_np)

    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for tif_path in tmp_dir.glob("*.tif"):
            zf.write(tif_path, arcname=tif_path.name)

    print(f"Wrote submission zip: {args.out}")


if __name__ == "__main__":
    main()
