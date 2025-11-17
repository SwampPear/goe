from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class GraphLayer(nn.Module):
    """
    Simple learnable message-passing layer over a fixed set of experts.

    Shapes
    ------
    h : (B, M, D)
        Expert states for a batch of size B, with M experts and hidden dim D.
    adjacency : (M, M), optional
        Row-normalized adjacency matrix A where A[m, n] is the weight for
        messages from expert n -> expert m. If None, a learned adjacency is
        used (self.edge_logits).

    Forward
    -------
    new_h = GraphLayer(h, adjacency)

    This implements:

        A = softmax(edge_logits, dim=-1)  # if adjacency is None
        messages_m = sum_n A[m, n] * h_n
        new_h_m = act(W(messages_m))

    where W is a shared linear projection.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        bias: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # Learnable edge logits: (M, M)
        self.edge_logits = nn.Parameter(torch.zeros(num_experts, num_experts))

        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, h: Tensor, adjacency: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        h : Tensor
            Expert states, shape (B, M, D).
        adjacency : Tensor, optional
            Row-normalized adjacency matrix, shape (M, M).

        Returns
        -------
        Tensor
            Updated expert states, shape (B, M, D).
        """
        if h.dim() != 3:
            raise ValueError(f"h must have shape (B, M, D), got {tuple(h.shape)}")

        B, M, D = h.shape
        if M != self.num_experts or D != self.hidden_dim:
            raise ValueError(
                f"Expected h.shape[1:] == ({self.num_experts}, {self.hidden_dim}), "
                f"got ({M}, {D})"
            )

        if adjacency is None:
            # (M, M) row-normalized
            A = torch.softmax(self.edge_logits, dim=-1)
        else:
            A = adjacency
            if A.shape != (self.num_experts, self.num_experts):
                raise ValueError(
                    f"adjacency must have shape ({self.num_experts}, {self.num_experts}), "
                    f"got {tuple(A.shape)}"
                )

        # Message passing: A[m, n] * h[:, n, :] -> messages[:, m, :]
        # A: (M, M), h: (B, M, D) => messages: (B, M, D)
        messages = torch.einsum("mn,bnd->bmd", A, h)

        out = self.proj(messages)
        out = self.act(out)
        return out


@dataclass
class AggregationOutput:
    """
    Convenience bundle for aggregation results.
    """

    expert_states: Tensor  # (B, M, D_e)
    routing_probs: Tensor  # (B, N, M)


def aggregate_tokens_to_experts(
    tokens: Tensor,
    routing_probs: Tensor,
    proj: Optional[nn.Linear] = None,
) -> Tensor:
    """
    Aggregate token embeddings into per-expert states via routing probabilities.

    This implements a differentiable version of equation (9):

        h_m = sum_i p_{i,m} * W_p * r_i

    Parameters
    ----------
    tokens : Tensor
        Token embeddings, shape (B, N, D).
    routing_probs : Tensor
        Routing distribution over experts, shape (B, N, M).
    proj : nn.Linear, optional
        Optional projection W_p applied to tokens before aggregation.
        If None, identity is used.

    Returns
    -------
    Tensor
        Expert states, shape (B, M, D_e) where D_e == proj.out_features or D.
    """
    if tokens.dim() != 3 or routing_probs.dim() != 3:
        raise ValueError(
            f"tokens and routing_probs must both be 3D, got "
            f"{tuple(tokens.shape)} and {tuple(routing_probs.shape)}"
        )

    B, N, D = tokens.shape
    Bp, Np, M = routing_probs.shape
    if B != Bp or N != Np:
        raise ValueError(
            f"Incompatible shapes: tokens (B={B}, N={N}), "
            f"routing_probs (B={Bp}, N={Np})"
        )

    if proj is not None:
        tokens = proj(tokens)  # (B, N, D_e)
    D_e = tokens.shape[-1]

    # Weighted sum over tokens: h_m = sum_i p_{i,m} * tokens_i
    # routing_probs: (B, N, M), tokens: (B, N, D_e)
    # -> expert_states: (B, M, D_e)
    expert_states = torch.einsum("bnm,bnd->bmd", routing_probs, tokens)
    assert expert_states.shape == (B, M, D_e)
    return expert_states


def scatter_experts_to_tokens(
    expert_states: Tensor,
    routing_probs: Tensor,
    proj: Optional[nn.Linear] = None,
) -> Tensor:
    """
    Distribute expert states back to tokens, weighted by routing probabilities.

    This is the approximate inverse of `aggregate_tokens_to_experts`:

        z_i = sum_m p_{i,m} * W_d * h_m

    Parameters
    ----------
    expert_states : Tensor
        Expert states, shape (B, M, D_e).
    routing_probs : Tensor
        Routing distribution over experts, shape (B, N, M).
    proj : nn.Linear, optional
        Optional projection W_d applied to expert states after aggregation.

    Returns
    -------
    Tensor
        Token embeddings reconstructed from expert states, shape (B, N, D_out).
    """
    if expert_states.dim() != 3 or routing_probs.dim() != 3:
        raise ValueError(
            f"expert_states and routing_probs must both be 3D, got "
            f"{tuple(expert_states.shape)} and {tuple(routing_probs.shape)}"
        )

    B, M, D_e = expert_states.shape
    Bp, N, Mp = routing_probs.shape
    if B != Bp or M != Mp:
        raise ValueError(
            f"Incompatible shapes: expert_states (B={B}, M={M}), "
            f"routing_probs (B={Bp}, M={Mp})"
        )

    # z_i = sum_m p_{i,m} * h_m
    # routing_probs: (B, N, M), expert_states: (B, M, D_e)
    # -> tokens: (B, N, D_e)
    tokens = torch.einsum("bnm,bmd->bnd", routing_probs, expert_states)

    if proj is not None:
        tokens = proj(tokens)

    return tokens
