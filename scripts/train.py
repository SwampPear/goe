from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from src.data.dataset import VesuviusPatchDataset
from src.data.losses import ink_loss
from src.utils.metrics import binary_dice

from scripts.common import build_model, match_target_to_logits, set_seed


def _parse_tuple3(arg: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in arg.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected format D,H,W (comma-separated).")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Graph-of-Experts (smoke test).")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-volumes", type=int, default=None)
    parser.add_argument("--patch-size", type=_parse_tuple3, default="32,64,64")
    parser.add_argument("--stride", type=_parse_tuple3, default="16,32,32")
    parser.add_argument("--stem-patch-size", type=_parse_tuple3, default="4,4,4")
    parser.add_argument("--stem-stride", type=_parse_tuple3, default="2,2,2")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny config for quick tests.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = VesuviusPatchDataset(
        root=args.data_root,
        split=args.split,
        patch_size=args.patch_size,
        stride=args.stride,
        augment=True,
        allow_missing_targets=False,
        limit_volumes=args.limit_volumes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = build_model(
        stem_patch_size=args.stem_patch_size,
        stem_stride=args.stem_stride,
        tiny=args.tiny,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            step += 1
            volume = batch["volume"].to(device)
            target = batch["target"].to(device)

            out = model(volume)
            logits = out["logits"]
            if logits.dim() != 5:
                raise RuntimeError(f"Expected 5D logits, got {tuple(logits.shape)}")

            target_resized = match_target_to_logits(target, logits)
            loss_main = ink_loss(logits, target_resized)
            loss_aux = out.get("aux_loss", torch.tensor(0.0, device=device))
            loss = loss_main + loss_aux

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                dice = binary_dice(logits, target_resized)
            if step % 5 == 0 or step == 1:
                print(f"step {step} loss={loss.item():.4f} dice={dice:.4f}")

            if step >= args.max_steps:
                break
        if step >= args.max_steps:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
