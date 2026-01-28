import argparse
import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.models.goe import GraphOfExperts, GraphOfExpertsConfig
from src.models.stem import InputStemConfig
from src.models.encoder import EncoderConfig
from src.models.router import GraphRouterConfig
from src.models.experts import ExpertsConfig
from src.models.decoder import DecoderConfig
from src.data.dataset import VesuviusPatchDataset
from src.utils import config as cfg


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    """Keeps track of running average of a metric (e.g., loss)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.cnt += n
        self.avg = self.sum / max(self.cnt, 1)


def save_checkpoint(state: Dict[str, Any], out_dir: Path, epoch: int, is_best: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(state, ckpt_path)
    if is_best:
        best_path = out_dir / "checkpoint_best.pt"
        torch.save(state, best_path)


def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Builds train/val dataloaders.

    You MUST adapt the VesuviusPatchDataset constructor to match your actual dataset API.
    """
    train_ds = VesuviusPatchDataset(
        root=args.data_root,
        split="train",
        patch_size=args.patch_size,
        stride=args.stride,
        augment=True,
        allow_missing_targets=False,
    )
    val_ds = VesuviusPatchDataset(
        root=args.data_root,
        split="val",
        patch_size=args.patch_size,
        stride=args.stride,
        augment=False,
        allow_missing_targets=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


# -----------------------
# Model / loss / optimizer
# -----------------------


def build_model(args, device: torch.device) -> nn.Module:
    """
    Instantiate Graph of Experts model.

    You may want to pass model hyperparameters from YAML via cfg.config(args.config, "model_*").
    Here we keep it simple and let the model define its own defaults.
    """
    cfg = GraphOfExpertsConfig(
        stem=InputStemConfig(
            patch_size=tuple(args.patch_size),
            patch_stride=tuple(args.stride),
        ),
        encoder=EncoderConfig(),
        router=GraphRouterConfig(),
        experts=ExpertsConfig(),
        decoder=DecoderConfig(),
    )
    model = GraphOfExperts(cfg)
    model.to(device)
    return model


def build_criterion(args) -> nn.Module:
    """
    Build loss function.

    For ink segmentation a common choice is BCEWithLogitsLoss; replace if you have a custom loss in src.data.losses.
    """
    # from src.data.losses import get_loss
    # return get_loss(args.loss_name)
    return nn.BCEWithLogitsLoss()


def build_optimizer(args, model: nn.Module):
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


# -----------------------
# Training / validation loops
# -----------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10,
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for step, batch in enumerate(loader):
        # You MUST adapt this unpacking to whatever your dataset returns:
        # Example assumption: (volume, target)
        volume = batch["volume"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass: GoE should consume volumetric x and output logits for ink (and/or geometry)
        out = model(volume)
        logits = out["logits"]  # shape: [B, 1, D, H, W] or similar

        loss = criterion(logits, target) + out.get("aux_loss", 0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item(), volume.size(0))

        if (step + 1) % log_interval == 0:
            print(
                f"[Epoch {epoch:03d} | Step {step + 1:05d}/{len(loader):05d}] "
                f"Loss: {loss_meter.avg:.4f}"
            )

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Returns (val_loss, dummy_metric).

    Replace dummy_metric with real metric(s) (e.g. IoU) if you have them in src.utils.metrics.
    """
    model.eval()
    loss_meter = AverageMeter()

    for batch in loader:
        volume = batch["volume"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        out = model(volume)
        logits = out["logits"]
        loss = criterion(logits, target) + out.get("aux_loss", 0.0)
        loss_meter.update(loss.item(), volume.size(0))

    # Placeholder metric: negative loss (so "higher is better" for checkpointing)
    metric = -loss_meter.avg
    return loss_meter.avg, metric


# -----------------------
# Main
# -----------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Graph of Experts on Vesuvius tomographic data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Name of YAML config in config/<name>.yaml (optional)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="goe_vesuvius",
        help="Name for logging / checkpoint directory",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/processed",
        help="Processed dataset root containing index.csv",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[16, 64, 64],
        metavar=("kD", "kH", "kW"),
        help="3D patch size used by the stem / dataset",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=[8, 32, 32],
        metavar=("sD", "sH", "sW"),
        help="3D stride used by the stem / dataset",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='"cuda", "cpu" or "cuda:0", etc.',
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs",
        help="Directory where checkpoints and logs are saved",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=1)

    args = parser.parse_args()

    # Optionally override from YAML config if provided
    if args.config is not None:
        try:
            yaml_seed = cfg.config(args.config, "seed")
            args.seed = yaml_seed
        except Exception:
            pass
        try:
            yaml_epochs = cfg.config(args.config, "epochs")
            args.epochs = yaml_epochs
        except Exception:
            pass
        try:
            yaml_batch = cfg.config(args.config, "batch_size")
            args.batch_size = yaml_batch
        except Exception:
            pass
        try:
            yaml_lr = cfg.config(args.config, "lr")
            args.lr = yaml_lr
        except Exception:
            pass
        try:
            yaml_exp = cfg.config(args.config, "experiment_name")
            args.experiment_name = yaml_exp
        except Exception:
            pass

    return args


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device
        if args.device != "cuda"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)
    print(f"Patch size: {patch_size}, stride: {stride}")

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args, device)
    criterion = build_criterion(args)
    optimizer = build_optimizer(args, model)

    out_dir = Path(args.out_dir) / args.experiment_name

    best_metric = -float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            log_interval=args.log_interval,
        )
        print(f"Epoch {epoch:03d} train loss: {train_loss:.4f}")

        if epoch % args.val_interval == 0:
            val_loss, metric = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch:03d} val loss: {val_loss:.4f} | metric (neg loss): {metric:.4f}"
            )

            is_best = metric > best_metric
            if is_best:
                best_metric = metric

            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "metric": metric,
                    "args": vars(args),
                },
                out_dir,
                epoch,
                is_best=is_best,
            )
        else:
            # still save a lightweight checkpoint every epoch
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "args": vars(args),
                },
                out_dir,
                epoch,
                is_best=False,
            )


if __name__ == "__main__":
    main()
