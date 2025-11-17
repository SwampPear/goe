from dataclasses import dataclass
from typing import Optional, Dict

import torch
from torch import nn, Tensor


@dataclass
class DecoderConfig:
    """
    Configuration for the graph-of-experts decoder.

    This module maps token-level representations and refined expert
    states back into volumetric predictions.

    Attributes
    ----------
    token_dim:
        Dimensionality of encoded tokens T_enc ∈ R^{B×N×d_token}.
    expert_state_dim:
        Dimensionality of expert states H ∈ R^{B×M×D_h}.
    num_experts:
        Number of experts M used by the router.
    out_channels:
        Number of output channels (e.g., 1 for ink logits, or K for
        multi-class segmentation).
    hidden_dim:
        Hidden size of the token-level decoder MLP.
    num_layers:
        Number of layers in the decoder MLP (≥ 1).
    dropout:
        Dropout probability in the decoder MLP.
    use_coords:
        If True, use normalized patch center coordinates from meta["coords"]
        as an additional input signal.
    """

    token_dim: int = 128
    expert_state_dim: int = 128
    num_experts: int = 4
    out_channels: int = 1
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    use_coords: bool = True


class DecoderMLP(nn.Module):
    """
    Generic MLP used to map per-token fused features → output logits.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be ≥ 1")

        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim

        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    Graph-of-Experts decoder.

    Given:
        - Encoded tokens T_enc ∈ R^{B×N×d_token}
        - Refined expert states H ∈ R^{B×M×D_h}
        - Optional routing probabilities p ∈ R^{B×N×M}
        - Optional meta (coords, grid_size, ...)

    it produces per-token logits y_i, which can be reshaped into a
    3D grid of shape [B, C_out, D_out, H_out, W_out] using the patch
    grid from the stem.

    Conditioning on experts is done via a mixture:
        c_i = Σ_m p_{i,m} W_e h_m
        z_i = concat( T_enc_i, c_i, optionally coord_i )
        y_i = f_dec(z_i)
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.cfg = cfg

        # Project expert states into token space before mixing per-token.
        self.expert_to_token = nn.Linear(cfg.expert_state_dim, cfg.token_dim)

        # Optional coordinate encoder (3-dim normalized coords → token_dim).
        if cfg.use_coords:
            self.coords_proj = nn.Linear(3, cfg.token_dim)
            coords_dim = cfg.token_dim
        else:
            self.coords_proj = None
            coords_dim = 0

        # Fused feature dimension: token + expert context (+ coords).
        fused_dim = cfg.token_dim + cfg.token_dim + coords_dim

        self.decoder_mlp = DecoderMLP(
            in_dim=fused_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_channels,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )

    def forward(
        self,
        tokens: Tensor,
        expert_states: Tensor,
        routing_probs: Optional[Tensor] = None,
        meta: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        tokens:
            Encoded tokens T_enc ∈ R^{B×N×d_token}.
        expert_states:
            Refined expert states H ∈ R^{B×M×D_h}.
        routing_probs:
            Optional routing probabilities p ∈ R^{B×N×M} from the router.
            If None, we fall back to uniform mixing over experts.
        meta:
            Optional dict with stem metadata:
                - "coords": [B, N, 3] normalized patch centers.
                - "grid_size": [3] tensor with (D_out, H_out, W_out).

        Returns
        -------
        logits:
            If meta["grid_size"] is provided:
                [B, C_out, D_out, H_out, W_out]
            otherwise:
                [B, N, C_out] (token-wise outputs).
        """
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected tokens of shape [B, N, d_token], got {tuple(tokens.shape)}"
            )
        if expert_states.dim() != 3:
            raise ValueError(
                f"Expected expert_states of shape [B, M, D_h], got {tuple(expert_states.shape)}"
            )

        B, N, d_token = tokens.shape
        B_h, M, D_h = expert_states.shape

        if B_h != B:
            raise ValueError(
                f"Batch size mismatch: tokens batch={B}, expert_states batch={B_h}"
            )
        if d_token != self.cfg.token_dim:
            raise ValueError(
                f"tokens last dim ({d_token}) must match cfg.token_dim ({self.cfg.token_dim})"
            )
        if D_h != self.cfg.expert_state_dim:
            raise ValueError(
                f"expert_states last dim ({D_h}) must match cfg.expert_state_dim ({self.cfg.expert_state_dim})"
            )
        if M != self.cfg.num_experts:
            raise ValueError(
                f"expert_states has M={M}, but cfg.num_experts={self.cfg.num_experts}"
            )

        # Routing probabilities: [B, N, M]
        if routing_probs is None:
            # Fallback: uniform expert mixture for each token.
            routing_probs = torch.full(
                (B, N, M),
                1.0 / float(M),
                device=tokens.device,
                dtype=tokens.dtype,
            )
        else:
            if routing_probs.shape != (B, N, M):
                raise ValueError(
                    f"Expected routing_probs of shape {(B, N, M)}, got {tuple(routing_probs.shape)}"
                )

        # Expert context per token:
        #   c_i = Σ_m p_{i,m} W_e h_m
        expert_states_tok = self.expert_to_token(expert_states)  # [B, M, token_dim]
        # routing_probs: [B, N, M]; expert_states_tok: [B, M, d_token]
        expert_context = torch.einsum("bnm,bmd->bnd", routing_probs, expert_states_tok)

        # Optional coordinate features
        if self.cfg.use_coords and meta is not None and "coords" in meta:
            coords = meta["coords"]
            if coords.shape != (B, N, 3):
                raise ValueError(
                    f"meta['coords'] must have shape {(B, N, 3)}, got {tuple(coords.shape)}"
                )
            coord_feats = self.coords_proj(coords)  # [B, N, token_dim]
        else:
            coord_feats = None

        # Fuse features: [tokens, expert_context, coord_feats?]
        if coord_feats is not None:
            fused = torch.cat([tokens, expert_context, coord_feats], dim=-1)
        else:
            fused = torch.cat([tokens, expert_context], dim=-1)

        # Per-token logits
        logits_token = self.decoder_mlp(fused)  # [B, N, C_out]

        # If grid_size is available, reshape to 3D grid; else return token-wise.
        if meta is not None and "grid_size" in meta:
            grid_size = meta["grid_size"]
            if grid_size.numel() != 3:
                raise ValueError(
                    f"meta['grid_size'] must have 3 elements (D, H, W), got {grid_size.numel()}"
                )
            D_out, H_out, W_out = map(int, grid_size.tolist())
            if D_out * H_out * W_out != N:
                raise ValueError(
                    f"grid_size product D*H*W={D_out * H_out * W_out} does not match N={N}"
                )

            logits_token = logits_token.view(B, D_out, H_out, W_out, self.cfg.out_channels)
            # [B, C_out, D_out, H_out, W_out]
            logits = logits_token.permute(0, 4, 1, 2, 3).contiguous()
            return logits

        # Fallback: no grid info → return token-wise logits [B, N, C_out]
        return logits_token
