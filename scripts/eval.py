from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import VesuviusPatchDataset
from src.utils.metrics import MetricCollection, binary_dice, binary_iou

from scripts.common import build_model, match_target_to_logits, set_seed


def _parse_tuple3(arg: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected format D,H,W (comma-separated).")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Graph-of-Experts (sanity metrics).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-volumes", type=int, default=None)
    parser.add_argument("--patch-size", type=_parse_tuple3, default="32,64,64")
    parser.add_argument("--stride", type=_parse_tuple3, default="16,32,32")
    parser.add_argument("--stem-patch-size", type=_parse_tuple3, default="4,4,4")
    parser.add_argument("--stem-stride", type=_parse_tuple3, default="2,2,2")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny config for quick tests.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = VesuviusPatchDataset(
        root=args.data_root,
        split=args.split,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=False,
        allow_missing_targets=False,
        limit_volumes=args.limit_volumes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.device.startswith("cuda"),
    )

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

    metrics = MetricCollection(["dice", "iou"])
    model.eval()
    with torch.no_grad():
        for batch in loader:
            volume = batch["volume"].to(device)
            target = batch["target"].to(device)
            out = model(volume)
            logits = out["logits"]
            if logits.dim() != 5:
                raise RuntimeError(f"Expected 5D logits, got {tuple(logits.shape)}")

            target_resized = match_target_to_logits(target, logits)
            metrics.update("dice", binary_dice(logits, target_resized), n=volume.size(0))
            metrics.update("iou", binary_iou(logits, target_resized), n=volume.size(0))

    scores = metrics.as_dict()
    print("Sanity metrics (not leaderboard metrics):")
    print(f"  Dice: {scores['dice']:.4f}")
    print(f"  IoU : {scores['iou']:.4f}")
    print("Leaderboard metrics not implemented: TopoScore, SurfaceDice@tau, VOI_score.")


if __name__ == "__main__":
    main()
