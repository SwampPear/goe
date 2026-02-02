from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.models.goe import GraphOfExperts, GraphOfExpertsConfig
from src.models.stem import InputStemConfig
from src.models.encoder import EncoderConfig
from src.models.router import GraphRouterConfig
from src.models.experts import ExpertsConfig
from src.models.decoder import DecoderConfig


@dataclass
class RunConfig:
    data_root: Path
    split: str
    device: str
    seed: int
    limit_volumes: Optional[int]
    batch_size: int
    num_workers: int
    patch_size: Tuple[int, int, int]
    stride: Tuple[int, int, int]
    stem_patch_size: Tuple[int, int, int]
    stem_stride: Tuple[int, int, int]
    tiny: bool


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_tuple3(vals: Sequence[int]) -> Tuple[int, int, int]:
    if len(vals) != 3:
        raise ValueError("Expected 3 integers.")
    return int(vals[0]), int(vals[1]), int(vals[2])


def build_model(
    stem_patch_size: Sequence[int],
    stem_stride: Sequence[int],
    tiny: bool = True,
) -> GraphOfExperts:
    if tiny:
        token_dim = 32
        aux_dim = 8
        num_heads = 4
        num_experts = 2
        stem = InputStemConfig(
            stem_channels=8,
            token_dim=token_dim,
            aux_dim=aux_dim,
            patch_size=_as_tuple3(stem_patch_size),
            patch_stride=_as_tuple3(stem_stride),
            use_layer_norm=True,
        )
        encoder = EncoderConfig(
            token_dim=token_dim,
            aux_dim=aux_dim,
            num_layers=1,
            num_heads=num_heads,
            ff_hidden_dim=64,
            dropout=0.0,
            use_positional_encoding=False,
            use_aux_update=False,
        )
        router = GraphRouterConfig(
            token_dim=token_dim,
            aux_dim=aux_dim,
            num_experts=num_experts,
            hidden_dim=32,
            expert_state_dim=token_dim,
            temperature=1.0,
            lambda_balance=0.0,
            lambda_entropy=0.0,
            use_aux=True,
        )
        experts = ExpertsConfig(
            num_experts=num_experts,
            state_dim=token_dim,
            hidden_dim=64,
            num_mlp_layers=1,
            message_passing_steps=1,
            dropout=0.0,
            shared_mlp=True,
            use_layer_norm=False,
        )
        decoder = DecoderConfig(
            token_dim=token_dim,
            expert_state_dim=token_dim,
            num_experts=num_experts,
            out_channels=1,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
            use_coords=True,
        )
    else:
        stem = InputStemConfig(
            patch_size=_as_tuple3(stem_patch_size),
            patch_stride=_as_tuple3(stem_stride),
        )
        encoder = EncoderConfig()
        router = GraphRouterConfig()
        experts = ExpertsConfig()
        decoder = DecoderConfig()

    cfg = GraphOfExpertsConfig(
        stem=stem,
        encoder=encoder,
        router=router,
        experts=experts,
        decoder=decoder,
    )
    return GraphOfExperts(cfg)


def match_target_to_logits(target: Tensor, logits: Tensor) -> Tensor:
    """
    Resize a target tensor to match logits spatial shape if needed.
    """
    if target.dim() == 4:
        target = target.unsqueeze(1)
    if target.shape[2:] != logits.shape[2:]:
        target = torch.nn.functional.interpolate(
            target.float(),
            size=logits.shape[2:],
            mode="nearest",
        )
    return target
