from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

# TODO: take out use_aux_update
# TODO: take out use_positional_encoding


@dataclass
class EncoderConfig:
    """
    Configuration for the token encoder and auxiliary refinement.
    Attributes:
        token_dim: size of each token embedding vector matching output of input stem's token projection
        aux_dim: dimensionality of auxiliary routing features produced by the stem
        num_layers: number of transformer encoder blocks stacked sequentially
        num_heads: number of attention heads in each multi-head self-attention layer, must divide token_dim evenly
        ff_hidden_dim: hidden layer size of each block's feed-forward network (typically 2x4xtoken_dim)
        dropout: dropout probability applied to attention, feed-forward activations, and residual connections
    """

    token_dim: int = 128
    aux_dim: int = 32
    num_layers: int = 4
    num_heads: int = 8
    ff_hidden_dim: int = 512
    dropout: float = 0.1


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding, broadcast over the batch. Tokens are treated as a 1D sequence in index
    space where 3D geometric structure is captured separately via coordinates and auxiliary routing features.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model) # [max_len, d_model], one embedding per position index

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # [max_len, 1]

        # geometric frequencies used for sin/cos pairs, matches Vaswani et al. (2017):
        #   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        #   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        # apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension so we can broadcast over [B, N, d_model].
        pe = pe.unsqueeze(0) # [1, max_len, d_model]

        # register_buffer ensures `pe` is moved with the module across devices
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for SinusiodalPositionalEncoding.
        Args:
            x: Tensor of shape [B, N, d_model] - input
        Returns:
            tokens with positional encoding added, same shape as x.
        """
        B, N, d = x.shape

        # guard against sequences longer than the precomputed max_len.
        if N > self.pe.size(1):
            raise ValueError(
                f"Sequence length {N} exceeds max_len {self.pe.size(1)} "
                "of positional encoding."
            )

        # slice positional encodings to current sequence length N and feature dim d
        return x + self.pe[:, :N, :d] # [B, N, d]


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
        Forward pass for MultiHeadAttention.
        Args:
            q: Tensor of shape [B, N_q, d_model] - query tensor
            k: Tensor of shape [B, N_k, d_model] - key tensor
            v: Tensor of shape [B, N_k, d_model] - value tensor
            key_padding_mask: Tensor of shape [B, N_k] - tensor where True indicates positions to ignore
        Returns:
            attention output [B, N_q, d_model]
        """
        B, Nq, _ = q.shape
        _, Nk, _ = k.shape

        q_proj = self._reshape_to_heads(self.q_proj(q))
        k_proj = self._reshape_to_heads(self.k_proj(k))
        v_proj = self._reshape_to_heads(self.v_proj(v))

        # scaled dot-product attention
        attn_weights = torch.matmul(q_proj, k_proj.transpose(-2, -1))
        attn_weights = attn_weights / (self.head_dim ** 0.5)

        if key_padding_mask is not None:
            # key_padding_mask: [B, Nk] -> [B, 1, 1, Nk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v_proj)      # [B, H, N_q, head_dim]
        attn_output = self._reshape_from_heads(attn_output) # [B, N_q, d_model]

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """
    Position-wise feedforward network:
        FF(x) = W2 alpha(W1 x + b1) + b2
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
    Auxiliary routing feature refinement implementing a lightweight cross-attention block where auxiliary routing 
    features act as queries over the contextual token embeddings. This keeps routing statistics aligned with the
    contextual embedding space produced by the encoder.
    """

    def __init__(self, token_dim: int, aux_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.token_dim = token_dim
        self.aux_dim = aux_dim

        self.aux_in = nn.Linear(aux_dim, token_dim)
        self.attn = MultiHeadAttention(token_dim, num_heads, dropout=dropout)
        self.aux_out = nn.Linear(token_dim, aux_dim)

        self.norm_aux = nn.LayerNorm(aux_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, aux: Tensor, tokens: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for AuxCrossAttentionUpdate.
        Args:
            aux: Tensor of shape [B, N, d_aux] - auxiliary routing features
            tokens: Tensor of shape [B, N, d_token] - contextual token embeddings
            key_padding_mask: optional Tensor of shape [B, N] - denotes which tokens should be processed
        Returns:
            updated auxiliary features [B, N, d_aux]
        """
        # normalize aux before projecting into token space
        aux_norm = self.norm_aux(aux)
        q = self.aux_in(aux_norm) # [B, N, token_dim]
        k = tokens
        v = tokens

        a_ctx = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        a_delta = self.aux_out(a_ctx)

        return aux + self.dropout(a_delta)


class EncoderLayer(nn.Module):
    """
    Single transformer encoder layer.

    Structure (pre-LN):
        x = x + MHA(LN(x))
        x = x + FF(LN(x))
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

    def forward(
        self,
        tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for EncoderLayer.
        Args:
            tokens: Tensor of shape [B, N, d_token] - input tokens
            key_padding_mask: optional Tensor of shape [B, N] - true marks padding
        Returns:
            updated tokens [B, N, d_token]
        """
        x = tokens

        # self-attention block
        x_norm = self.norm1(x)
        attn_out = self.self_attn(
            q=x_norm,
            k=x_norm,
            v=x_norm,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(attn_out)

        # feedforward block
        x_norm2 = self.norm2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)

        return x


class Encoder(nn.Module):
    """
    Token encoder auxiliary refinement implementing a stack of transformer-style self-attention blocks operating
    purely in token space, while auxiliary refinment is a lightweight cross-attention module that refines routing 
    features using the contextual tokens.
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

        self.aux_update: Optional[AuxCrossAttentionUpdate] = None
        if cfg.use_aux_update:
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
        Forward pass for Encoder.
        Args:
            tokens: Tensor of shape [B, N, d_token] - input token embeddings
            aux: Tensor of shape [B, N, d_aux] - input token embeddings
            key_padding_mask: optional Tensor of shape [B, N] - true marks invalid tokens

        Returns:
            (tokens_out, aux_out)
            tokens_out: contextualized token embeddings [B, N, d_token]
            aux_out: contextualized token embeddings [B, N, d_aux]
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

        # optional positional encoding in token space.
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # transformer encoder stack: T → T′
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        # optional auxiliary refinement: a → a′ using T′ as context.
        a_out: Optional[Tensor] = aux
        if (
            self.aux_update is not None
            and aux is not None
            and self.cfg.use_aux_update
        ):
            a_out = self.aux_update(aux, x, key_padding_mask=key_padding_mask)

        return x, a_out
