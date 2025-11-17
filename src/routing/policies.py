from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class RoutingPolicy(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for routing policies that convert per-token logits to
    per-token routing probabilities over experts.

    Forward signature
    -----------------
    probs = policy(logits, mask=None)

    Parameters
    ----------
    logits : Tensor
        Unnormalized routing scores, shape (B, N, M).
    mask : Tensor, optional
        Binary mask over tokens, shape (B, N). 0 marks padded tokens and
        will zero-out their probabilities.
    """

    @abc.abstractmethod
    def forward(self, logits: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class SoftmaxRouting(RoutingPolicy):
    """
    Standard softmax gating over experts.

    This is the default "dense" router: every token is softly assigned
    to all experts according to a temperature-controlled softmax.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, logits: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits must have shape (B, N, M), got {tuple(logits.shape)}")

        scores = logits / self.temperature

        if mask is not None:
            if mask.shape != logits.shape[:2]:
                raise ValueError(
                    f"mask must have shape (B, N), got {tuple(mask.shape)} vs {tuple(logits.shape)}"
                )
            # Set masked tokens to -inf so softmax -> 0 prob.
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))

        probs = torch.softmax(scores, dim=-1)

        # If mask is given, ensure masked positions are exactly zero.
        if mask is not None:
            probs = probs * mask.unsqueeze(-1)

        return probs


class TopKRouting(RoutingPolicy):
    """
    Top-k sparse gating.

    Keeps only the top-k experts per token and zeroes out the rest,
    followed by renormalization over the selected experts.
    """

    def __init__(self, k: int = 2, temperature: float = 1.0) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be > 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.k = int(k)
        self.temperature = float(temperature)

    def forward(self, logits: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits must have shape (B, N, M), got {tuple(logits.shape)}")

        B, N, M = logits.shape
        k = min(self.k, M)

        scores = logits / self.temperature

        if mask is not None:
            if mask.shape != logits.shape[:2]:
                raise ValueError(
                    f"mask must have shape (B, N), got {tuple(mask.shape)} vs {tuple(logits.shape)}"
                )
            scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))

        # Top-k over experts
        topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)  # (B, N, k)

        # Build sparse scores tensor with -inf for non-topk.
        sparse_scores = torch.full_like(scores, float("-inf"))
        sparse_scores.scatter_(-1, topk_idx, topk_vals)

        probs = torch.softmax(sparse_scores, dim=-1)

        if mask is not None:
            probs = probs * mask.unsqueeze(-1)

        return probs


@dataclass
class RecursiveState:
    """
    Small helper for recursive / iterative routing.

    Attributes
    ----------
    step : int
        Current recursion depth (0-based).
    confidence : Tensor
        Per-token confidence scores, shape (B, N).
    done_mask : Tensor
        Binary mask, shape (B, N), 1 where recursion has stopped.
    """

    step: int
    confidence: Tensor
    done_mask: Tensor


class RecursiveRouter(nn.Module):
    """
    Simple controller for recursive / early-stopping routing.

    This module does NOT perform routing itself; instead, it consumes a
    per-token confidence estimate over time and produces a "done_mask"
    indicating which tokens should stop routing.

    The intended usage is inside a loop:

        state = None
        for step in range(max_steps):
            logits, confidence = router_layer(...)
            probs = routing_policy(logits, mask=~state.done_mask if state else None)
            state = rec_router.update(confidence, state)
            if state.done_mask.all():
                break

    Stopping rule
    -------------
    A token is marked as done when:

        step >= min_steps
        AND |confidence_t - confidence_{t-1}| < tol

    or when max_steps is reached.
    """

    def __init__(
        self,
        max_steps: int = 3,
        min_steps: int = 1,
        tol: float = 1e-3,
    ) -> None:
        super().__init__()
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if min_steps < 0 or min_steps > max_steps:
            raise ValueError("0 <= min_steps <= max_steps must hold")
        if tol < 0:
            raise ValueError("tol must be >= 0")

        self.max_steps = int(max_steps)
        self.min_steps = int(min_steps)
        self.tol = float(tol)

    def update(self, confidence: Tensor, prev: Optional[RecursiveState]) -> RecursiveState:
        """
        Parameters
        ----------
        confidence : Tensor
            Per-token
