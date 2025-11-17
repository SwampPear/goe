from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor


@dataclass
class ExpertsConfig:
    """
    Configuration for the Graph-of-Experts refinement block.

    Attributes
    ----------
    num_experts:
        Number of expert nodes |V| in the graph.
    state_dim:
        Dimensionality of expert states h_m. Should match router.expert_state_dim.
    hidden_dim:
        Hidden size of the per-expert MLP(s).
    num_mlp_layers:
        Number of layers in each expert MLP (≥ 1).
    message_passing_steps:
        Number of GNN-style message passing steps over the expert graph.
    dropout:
        Dropout probability for MLP and message passing.
    shared_mlp:
        If True, all experts share a single MLP; if False, each expert has its
        own parameters (one MLP per node).
    use_layer_norm:
        Whether to apply LayerNorm to expert states after each message passing step.
    """

    num_experts: int = 4
    state_dim: int = 128
    hidden_dim: int = 256
    num_mlp_layers: int = 2
    message_passing_steps: int = 2
    dropout: float = 0.1
    shared_mlp: bool = False
    use_layer_norm: bool = True


class ExpertMLP(nn.Module):
    """
    Simple MLP used inside each expert node:

        h' = MLP(h)

    The exact functional form is:
        Linear -> GELU -> Dropout -> (repeated) -> Linear
    """

    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")

        layers = []
        in_dim = state_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        # Final projection back to state_dim
        layers.append(nn.Linear(in_dim, state_dim))
        layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, h: Tensor) -> Tensor:
        """
        Parameters
        ----------
        h:
            Expert state tensor of shape [..., D_h].

        Returns
        -------
        h_out:
            Updated expert state tensor of same shape.
        """
        return self.net(h)


class Experts(nn.Module):
    """
    Graph-of-Experts refinement module.

    This implements Eq. (10)-style refinement over expert states:

        1. Per-expert nonlinearity:
           h̃_m = f_m(h_m)                             (expert MLP)

        2. Graph message passing using adjacency A:
           m_m = Σ_n A_{mn} W_msg h̃_n
           h_m^{new} = h_m + h̃_m + m_m                (residual update)

    Repeating this for `message_passing_steps` steps yields refined expert
    states that have communicated via the learned expert graph.
    """

    def __init__(self, cfg: ExpertsConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.shared_mlp:
            # One shared MLP for all experts
            self.shared_mlp = ExpertMLP(
                state_dim=cfg.state_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_mlp_layers,
                dropout=cfg.dropout,
            )
            self.expert_mlps: Optional[nn.ModuleList] = None
        else:
            # One MLP per expert (independent parameters)
            self.shared_mlp = None
            self.expert_mlps = nn.ModuleList(
                [
                    ExpertMLP(
                        state_dim=cfg.state_dim,
                        hidden_dim=cfg.hidden_dim,
                        num_layers=cfg.num_mlp_layers,
                        dropout=cfg.dropout,
                    )
                    for _ in range(cfg.num_experts)
                ]
            )

        # Message passing projection W_msg
        self.msg_proj = nn.Linear(cfg.state_dim, cfg.state_dim)
        self.msg_dropout = nn.Dropout(cfg.dropout)

        # Optional LayerNorm after each step
        self.norm = nn.LayerNorm(cfg.state_dim) if cfg.use_layer_norm else nn.Identity()

    def _apply_expert_mlp(self, h: Tensor) -> Tensor:
        """
        Apply either a shared MLP or per-expert MLPs over expert states.

        Parameters
        ----------
        h:
            Expert states of shape [B, M, D_h].

        Returns
        -------
        h_tilde:
            Updated states of shape [B, M, D_h].
        """
        B, M, D = h.shape
        if M != self.cfg.num_experts:
            raise ValueError(
                f"Expected {self.cfg.num_experts} experts, got {M} in expert_states."
            )

        if self.shared_mlp is not None:
            # Shared: just apply once to the entire tensor
            return self.shared_mlp(h)
        else:
            # Per-expert MLPs: loop over M (M is typically small)
            assert self.expert_mlps is not None
            out = torch.empty_like(h)
            for m in range(M):
                out[:, m, :] = self.expert_mlps[m](h[:, m, :])
            return out

    def forward(self, expert_states: Tensor, adjacency: Tensor) -> Tensor:
        """
        Parameters
        ----------
        expert_states:
            Initial expert states h̄_m from the router,
            shape [B, M, D_h].
        adjacency:
            Row-normalized expert adjacency A_{mn},
            shape [M, M], where rows index source experts m and
            columns index destination n (or vice versa depending on
            your convention; here we treat it as outgoing edges from m
            to n and use an einsum that is consistent with that).

        Returns
        -------
        refined_states:
            Refined expert states h_m^{(T)} after T = message_passing_steps
            rounds of per-expert updates and graph message passing,
            shape [B, M, D_h].
        """
        if expert_states.dim() != 3:
            raise ValueError(
                f"Expected expert_states of shape [B, M, D_h], "
                f"got {tuple(expert_states.shape)}"
            )
        if adjacency.dim() != 2:
            raise ValueError(
                f"Expected adjacency of shape [M, M], got {tuple(adjacency.shape)}"
            )

        B, M, D = expert_states.shape
        if M != self.cfg.num_experts:
            raise ValueError(
                f"expert_states has M={M}, but cfg.num_experts={self.cfg.num_experts}"
            )
        if adjacency.shape != (M, M):
            raise ValueError(
                f"Expected adjacency of shape ({M}, {M}), "
                f"got {tuple(adjacency.shape)}"
            )

        # Optionally, you could re-normalize adjacency here to be safe:
        # adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        h = expert_states
        for _ in range(self.cfg.message_passing_steps):
            # 1) Per-expert nonlinearity
            h_tilde = self._apply_expert_mlp(h)  # [B, M, D]

            # 2) Graph message passing
            # messages_m = Σ_n A_{mn} W_msg(h̃_n)
            h_msg_in = self.msg_proj(h_tilde)  # [B, M, D]
            # adjacency: [M, M], h_msg_in: [B, M, D]
            # We want: [B, M, D] where each m aggregates from all n.
            messages = torch.einsum("mn,bnd->bmd", adjacency, h_msg_in)

            messages = self.msg_dropout(messages)

            # 3) Residual update + optional normalization
            h = h + h_tilde + messages
            h = self.norm(h)

        return h
