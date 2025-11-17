from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def _masked_probs(
    probs: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Ensure probs are zero on masked tokens and return the masked tensor.

    probs : (B, N, M)
    mask  : (B, N), optional
    """
    if mask is None:
        return probs

    if mask.shape != probs.shape[:2]:
        raise ValueError(
            f"mask must have shape (B, N) compatible with probs (B, N, M), "
            f"got {tuple(mask.shape)} vs {tuple(probs.shape)}"
        )

    return probs * mask.unsqueeze(-1)


def load_balance_loss(
    routing_probs: Tensor,
    mask: Optional[Tensor] = None,
    lambda_balance: float = 1.0,
) -> Tensor:
    """
    Load-balancing regularizer over experts.

    Encourages the mean routing probability per expert to be close to 1 / M,
    where M is the number of experts. This is a differentiable version of
    equation (11) in the paper:

        L_balance = λ_b * ( (1/B) sum_i p_{i,m} - 1/M )^2

    Parameters
    ----------
    routing_probs : Tensor
        Routing distribution over experts, shape (B, N, M).
    mask : Tensor, optional
        Binary mask over tokens, shape (B, N). 0 marks padded tokens.
    lambda_balance : float
        Scaling factor λ_b.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    if routing_probs.dim() != 3:
        raise ValueError(
            f"
