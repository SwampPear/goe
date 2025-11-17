from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn, Tensor


@dataclass
class GraphRouterConfig:
    """
    Configuration for the GraphRouter.

    Attributes
    ----------
    token_dim:
        Dimensionality of contextual token embeddings T' ∈ R^{B×N×d_token}.
    aux_dim:
        Dimensionality of auxiliary routing descriptors a' ∈ R^{B×N×d_aux}.
    num_experts:
        Number of expert nodes |V| in the graph G = (V, E).
    hidden_dim:
        Hidden size of the router MLP that maps [t'_i; a'_i] to logits.
    expert_state_dim:
        Dimensionality of expert states h_m. If None, defaults to token_dim.
    temperature:
        Optional softmax temperature for the routing distribution p_{i,m}.
    lambda_balance:
        Coefficient for the load-balancing regularizer L_balance.
    lambda_entropy:
        Coefficient for the entropy regularizer L_entropy.
    use_aux:
        If True, concatenates tokens and aux for routing; otherwise uses tokens only.
    """

    token_dim: int = 128
    aux_dim: int = 32
    num_experts: int = 4
    hidden_dim: int = 128
    expert_state_dim: Optional[int] = None  # defaults to token_dim if None
    temperature: float = 1.0
    lambda_balance: float = 1e-2
    lambda_entropy: float = 1e-3
    use_aux: bool = True


class GraphRouter(nn.Module):
    """
    Graph router Φ_r that turns token-level features into:

        • Routing probabilities p_{i,m} over experts m ∈ {1..M}.
        • Expert-initial states  h̄_m  via differentiable aggregation (Eq. 9).
        • A learnable expert adjacency matrix A_{mn} (row-normalized).

    This module does *not* implement the expert refinement itself (Eq. 10);
    it just produces the routing structure and regularization terms that
    will be consumed by the expert blocks.
    """

    def __init__(self, cfg: GraphRouterConfig):
        super().__init__()
        self.cfg = cfg

        expert_state_dim = cfg.expert_state_dim or cfg.token_dim
        self.expert_state_dim = expert_state_dim

        # Input to the router MLP is [t'_i; a'_i] if use_aux, else just t'_i.
        in_dim = cfg.token_dim + (cfg.aux_dim if cfg.use_aux else 0)

        # Token → expert logits (p_i,m before softmax).
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.num_experts),
        )

        # Projection W_p used when aggregating tokens into expert states (Eq. 9).
        self.proj_to_state = nn.Linear(in_dim, expert_state_dim)

        # Global learnable adjacency logits Ã ∈ R^{M×M}.
        # Softmax over the last dimension gives row-normalized A.
        self.adj_logits = nn.Parameter(torch.empty(cfg.num_experts, cfg.num_experts))
        nn.init.xavier_uniform_(self.adj_logits)

    def forward(
        self,
        tokens: Tensor,
        aux: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        tokens:
            Contextual token embeddings T' ∈ R^{B×N×d_token}.
        aux:
            Auxiliary routing descriptors a' ∈ R^{B×N×d_aux}, or None if
            cfg.use_aux = False.
        key_padding_mask:
            Optional mask of shape [B, N] where True indicates padding tokens
            that should be ignored for routing statistics and aggregation.

        Returns
        -------
        out:
            Dict with:
                - "logits": [B, N, M] routing logits over experts.
                - "probs": [B, N, M] routing probabilities p_{i,m}.
                - "expert_states": [B, M, D_h] aggregated expert states h̄_m.
                - "adjacency": [M, M] row-normalized expert adjacency A_{mn}.
                - "load_balance_loss": scalar tensor (L_balance).
                - "entropy_loss": scalar tensor (L_entropy).
        """
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected tokens of shape [B, N, d_token], got {tuple(tokens.shape)}"
            )

        B, N, d = tokens.shape
        if d != self.cfg.token_dim:
            raise ValueError(
                f"Last dim of tokens ({d}) must match cfg.token_dim "
                f"({self.cfg.token_dim})."
            )

        # Build the routing representation r_i = [t'_i; a'_i].
        if self.cfg.use_aux:
            if aux is None:
                raise ValueError("cfg.use_aux is True but aux is None.")
            if aux.shape != (B, N, self.cfg.aux_dim):
                raise ValueError(
                    f"Expected aux of shape {(B, N, self.cfg.aux_dim)}, "
                    f"got {tuple(aux.shape)}"
                )
            r = torch.cat([tokens, aux], dim=-1)  # [B, N, d_token + d_aux]
        else:
            r = tokens  # [B, N, d_token]

        # Router logits over experts
        logits = self.router_mlp(r)  # [B, N, M]
        if self.cfg.temperature != 1.0:
            logits = logits / self.cfg.temperature

        # Optionally mask out padding tokens (set to -inf so softmax → 0).
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, N):
                raise ValueError(
                    f"Expected key_padding_mask of shape {(B, N)}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            mask = key_padding_mask.unsqueeze(-1)  # [B, N, 1]
            logits = logits.masked_fill(mask, float("-inf"))

        probs = torch.softmax(logits, dim=-1)  # [B, N, M]

        # Build a "valid" mask in float form for aggregation/statistics.
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).to(probs.dtype)  # [B, N, 1]
        else:
            valid_mask = torch.ones(B, N, 1, device=tokens.device, dtype=probs.dtype)

        # Zero out probabilities for padded tokens for stats/aggregation.
        probs_valid = probs * valid_mask  # [B, N, M]

        # Project token-level features into the expert state space.
        r_state = self.proj_to_state(r)  # [B, N, D_h]

        # Aggregate tokens into expert states (Eq. 9):
        #   h̄_m = Σ_i p_{i,m} W_p r_i  /  Σ_i p_{i,m}
        # Use einsum: [B, N, D_h] × [B, N, M] → [B, M, D_h]
        h_num = torch.einsum("bnd,bnm->bmd", r_state, probs_valid)  # numerator

        mass = probs_valid.sum(dim=1)  # [B, M] total routing mass per expert
        denom = mass.unsqueeze(-1).clamp_min(1e-6)  # [B, M, 1]
        expert_states = h_num / denom  # [B, M, D_h]

        # Row-normalized adjacency A_{mn}: softmax over outgoing edges.
        adjacency = torch.softmax(self.adj_logits, dim=-1)  # [M, M]

        # -----------------------------
        # Regularizers: L_balance, L_entropy
        # -----------------------------
        # Load-balancing: encourage each expert to receive ~equal mass.
        device = tokens.device
        load_balance_loss = torch.tensor(0.0, device=device)
        if self.cfg.lambda_balance != 0.0:
            # Sum over tokens → [B, M]
            sum_over_tokens = probs_valid.sum(dim=1)  # [B, M]

            # Number of valid tokens per batch element
            valid_tokens_per_batch = valid_mask.squeeze(-1).sum(dim=1).clamp_min(1.0)  # [B]

            # Average per batch, then average across batch → [M]
            mean_per_batch = sum_over_tokens / valid_tokens_per_batch.unsqueeze(-1)  # [B, M]
            mean_p = mean_per_batch.mean(dim=0)  # [M]

            target = 1.0 / self.cfg.num_experts
            load_balance_loss = ((mean_p - target) ** 2).mean() * self.cfg.lambda_balance

        # Entropy regularizer: encourage high-entropy routing distributions.
        entropy_loss = torch.tensor(0.0, device=device)
        if self.cfg.lambda_entropy != 0.0:
            p_clamped = probs.clamp_min(1e-8)
            entropy = -(p_clamped * p_clamped.log()).sum(dim=-1)  # [B, N]

            if key_padding_mask is not None:
                entropy = entropy * (~key_padding_mask).to(entropy.dtype)
                num_valid = valid_mask.squeeze(-1).sum(dim=1).clamp_min(1.0)  # [B]
                mean_entropy = (entropy.sum(dim=1) / num_valid).mean()
            else:
                mean_entropy = entropy.mean()

            # L_entropy = -λ_e * H̄, so minimizing L encourages larger entropy.
            entropy_loss = -self.cfg.lambda_entropy * mean_entropy

        return {
            "logits": logits,                   # [B, N, M]
            "probs": probs,                     # [B, N, M]
            "expert_states": expert_states,     # [B, M, D_h]
            "adjacency": adjacency,             # [M, M]
            "load_balance_loss": load_balance_loss,  # scalar
            "entropy_loss": entropy_loss,            # scalar
        }
