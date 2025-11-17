from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor


@dataclass
class EncoderConfig:
    """
    Configuration for the token encoder Φ(T; θ_e).

    Attributes
    ----------
    token_dim:
        Embedding dimension of tokens T ∈ R^{B×N×d}.
    aux_dim:
        Dimensionality of auxiliary routing features a ∈ R^{B×N×d_aux}.
    num_layers:
        Number of transformer encoder layers.
    num_heads:
        Number of attention heads in multi-head attention.
    ff_hidden_dim:
        Hidden dimension of the feedforward network in each layer.
    dropout:
        Dropout probability applied to attention and feedforward outputs.
    use_positional_encoding:
        Whether to apply sinusoidal positional encoding to the tokens.
    use_aux_update:
        Whether to refine auxiliary features via f_aux(a, T′).
    """

    token_dim: int = 128
    aux_dim: int = 32
    num_layers: int = 4
    num_heads: int = 8
    ff_hidden_dim: int = 512
    dropout: float = 0.1
    use_positional_encoding: bool = True
    use_aux_update: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding, broadcast over the batch.

    This treats the token sequence as 1D in index space; 3D geometric
    structure is captured separately via coords/aux features.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x:
            Token tensor of shape [B, N, d_model].

        Returns
        -------
        x_pe:
            Tokens with positional encoding added, same shape as x.
        """
        B, N, d = x.shape
        if N > self.pe.size(1):
            raise ValueError(
                f"Sequence length {N} exceeds max_len {self.pe.size(1)} "
                "of positional encoding."
            )
        return x + self.pe[:, :N, :d]


class MultiHeadAttention(nn.Module):
    """
    Generic multi-head attention:

        Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Supports both self-attention (Q = K = V) and cross-attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_heads(self, x: Tensor) -> Tensor:
        # [B, N, d_model] -> [B, num_heads, N, head_dim]
        B, N, _ = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _reshape_from_heads(self, x: Tensor) -> Tensor:
        # [B, num_heads, N, head_dim] -> [B, N, d_model]
        B, H, N, Hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(B, N, H * Hd)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        q:
            Query tensor [B, N_q, d_model].
        k:
            Key tensor [B, N_k, d_model].
        v:
            Value tensor [B, N_k, d_model].
        key_padding_mask:
            Optional mask [B, N_k] where True indicates positions to ignore.

        Returns
        -------
        out:
            Attention output [B, N_q, d_model].
        """
        B, Nq, _ = q.shape
        _, Nk, _ = k.shape

        q_proj = self._reshape_to_heads(self.q_proj(q))
        k_proj = self._reshape_to_heads(self.k_proj(k))
        v_proj = self._reshape_to_heads(self.v_proj(v))

        # Scaled dot-product attention
        attn_weights = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            # key_padding_mask: [B, Nk] -> [B, 1, 1, Nk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v_proj)  # [B, H, N_q, head_dim]
        attn_output = self._reshape_from_heads(attn_output)  # [B, N_q, d_model]

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """
    Position-wise feedforward network:

        FF(x) = W2 σ(W1 x + b1) + b2
    """

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AuxCrossAttentionUpdate(nn.Module):
    """
    f_aux(a, T′): auxiliary routing feature refinement.

    Implements a light cross-attention block where auxiliary features
    act as queries over the context-aware token embeddings:

        a_proj = W_a_in a
        a_hat = Attn(a_proj, T′, T′)
        a′ = W_a_out a_hat + a

    This aligns routing statistics with the evolving contextual embedding
    space of the encoder.
    """

    def __init__(
        self,
        token_dim: int,
        aux_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.aux_dim = aux_dim

        self.aux_in = nn.Linear(aux_dim, token_dim)
        self.attn = MultiHeadAttention(token_dim, num_heads, dropout=dropout)
        self.aux_out = nn.Linear(token_dim, aux_dim)

        self.norm_aux = nn.LayerNorm(aux_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        aux: Tensor,
        tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        aux:
            Auxiliary routing features a ∈ R^{B×N×d_aux}.
        tokens:
            Contextual token embeddings T′ ∈ R^{B×N×d_token}.
        key_padding_mask:
            Optional token padding mask [B, N].

        Returns
        -------
        a_prime:
            Updated auxiliary features a′ ∈ R^{B×N×d_aux}.
        """
        # Normalize aux before projecting into token space
        aux_norm = self.norm_aux(aux)
        q = self.aux_in(aux_norm)  # [B, N, token_dim]
        k = tokens
        v = tokens

        a_ctx = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        a_delta = self.aux_out(a_ctx)
        return aux + self.dropout(a_delta)


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer with optional auxiliary update.

    Structure:
        x = x + MHA(LN(x))
        x = x + FF(LN(x))
        a = f_aux(a, x)    (if enabled)
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d_model = cfg.token_dim

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(
            d_model=d_model,
            hidden_dim=cfg.ff_hidden_dim,
            dropout=cfg.dropout,
        )

        self.dropout = nn.Dropout(cfg.dropout)

        self.use_aux_update = cfg.use_aux_update
        self.aux_update: Optional[AuxCrossAttentionUpdate] = None
        if self.use_aux_update:
            self.aux_update = AuxCrossAttentionUpdate(
                token_dim=cfg.token_dim,
                aux_dim=cfg.aux_dim,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )

    def forward(
        self,
        tokens: Tensor,
        aux: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        tokens:
            Input tokens T ∈ R^{B×N×d_token}.
        aux:
            Auxiliary features a ∈ R^{B×N×d_aux}, or None.
        key_padding_mask:
            Optional mask over tokens [B, N], where True marks padding.

        Returns
        -------
        tokens_out:
            Updated tokens T′ ∈ R^{B×N×d_token}.
        aux_out:
            Updated auxiliary features a′ ∈ R^{B×N×d_aux}, or None.
        """
        # Self-attention block
        x = tokens
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            q=x_norm,
            k=x_norm,
            v=x_norm,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # Feedforward block
        x_norm2 = self.norm2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)

        # Auxiliary update (optional)
        if self.use_aux_update and aux is not None and self.aux_update is not None:
            aux = self.aux_update(aux, x, key_padding_mask=key_padding_mask)

        return x, aux


class Encoder(nn.Module):
    """
    Token encoder Φ(T; θ_e) with optional auxiliary refinement.

    Implements Section 2.3:

        T′ = Φ(T; θ_e)                (3)
        a′ = f_aux(a, T′)             (5)

    where Φ is a stack of transformer-style self-attention blocks operating
    in token space, and f_aux is a lightweight cross-attention update that
    aligns routing statistics with the contextualized tokens.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.num_layers)]
        )

        self.pos_encoding: Optional[SinusoidalPositionalEncoding] = None
        if cfg.use_positional_encoding:
            self.pos_encoding = SinusoidalPositionalEncoding(cfg.token_dim)

    def forward(
        self,
        tokens: Tensor,
        aux: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        tokens:
            Input token embeddings T ∈ R^{B×N×d_token}.
        aux:
            Auxiliary routing features a ∈ R^{B×N×d_aux}, or None.
        key_padding_mask:
            Optional padding mask [B, N] where True marks invalid positions.

        Returns
        -------
        tokens_out:
            Contextualized token embeddings T′ ∈ R^{B×N×d_token}.
        aux_out:
            Updated auxiliary features a′ ∈ R^{B×N×d_aux}, or None.
        """
        if tokens.dim() != 3:
            raise ValueError(
                f"Expected tokens of shape [B, N, d_token], got {tuple(tokens.shape)}"
            )

        if aux is not None and aux.dim() != 3:
            raise ValueError(
                f"Expected aux of shape [B, N, d_aux], got {tuple(aux.shape)}"
            )

        B, N, d = tokens.shape
        if d != self.cfg.token_dim:
            raise ValueError(
                f"Last dim of tokens ({d}) must match cfg.token_dim "
                f"({self.cfg.token_dim})."
            )

        if aux is not None and aux.shape[:2] != (B, N):
            raise ValueError(
                "Auxiliary features must share [B, N] dimensions with tokens"
            )

        x = tokens

        # Optional positional encoding in token space.
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        a = aux
        for layer in self.layers:
            x, a = layer(x, a, key_padding_mask=key_padding_mask)

        return x, a
