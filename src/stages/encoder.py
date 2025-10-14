import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple

class TransformerBlock(layers.Layer):
    """
    A pre-norm Transformer encoder block: LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual.

    Args:
        d_model: token embedding width (must be divisible by num_heads)
        num_heads: number of attention heads
        ff_dim: hidden size of the feedforward network
        dropout: dropout rate applied to attention and feedforward network outputs
    """
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            attention_dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        """
        Apply MHSA and FFN with pre-norm and residual connections.

        Args:
            x: input tokens of shape [B, N, d_model]
            mask: optional attention mask of shape [B, 1, N] or [B, N] (True/1 = keep, False/0 = pad)
            training: enables dropout when True

        Returns:
            tensor of shape [B, N, d_model], same dtype as `x`
        """
        # pre-norm -> feedforward network
        y = self.norm1(x)
        attn_output = self.attn(y, y, attention_mask=mask, training=training)
        x = x + self.drop(attn_output, training=training)  # residual 1

        # pre-norm -> feedforward network
        y = self.norm2(x)
        ffn_output = self.ffn(y, training=training)
        x = x + self.drop(ffn_output, training=training)   # residual 2

        return x


class Encoder(layers.Layer):
    """
    Token encoder with stacked Transformer blocks and auxiliary-feature refinement via cross-attention.

    Pipeline:
      1) Encode tokens: T -> T' using L stacked TransformerBlock layers
      2) Refine auxiliary features: a -> a' by cross-attending a (projected) over T'

    Args:
        d_model: token width for T (and output)
        num_heads: number of heads for MHSA/cross-attention
        ff_dim: feedforwardd network hidden size inside each TransformerBlock
        num_layers: number of TransformerBlock layers
        d_aux: width of auxiliary features (input and output a, a')
        dropout: dropout rate for attention and feedforward network
    """
    def __init__(self, d_model: int = 256, num_heads: int = 8, ff_dim: int = 512, num_layers: int = 4, d_aux: int = 64, 
        dropout: float = 0.1):
        super().__init__()
        self.encoder_layers = [
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]

        self.aux_proj = layers.Dense(d_aux)
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            attention_dropout=dropout
        )

        self.norm_aux1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm_aux2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, T, a, mask=None, training=False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Run the encoder.

        Steps:
          (1) T'   = phi(T; theta_e) via stacked Transformer blocks with shared mask
          (2) a_t  = LN(Dense(a)) # normalize and project a
          (3) a_h  = CrossAttn(query=a_t, key=T', value=T', mask)
          (4) a'   = LN(a_t + Dropout(aa_h)) # residual + norm

        Args:
            T: token tensor [B, N, d_model]
            a: auxiliary features [B, N, d_aux]
            mask: optional padding mask [B, 1, N] or [B, N] (True/1 = keep)
            training: enables dropout when True

        Returns:
            (T_prime, a_prime) with shapes [B, N, d_model], [B, N, d_aux]
        """
        for block in self.encoder_layers:
            T = block(T, mask=mask, training=training)

        a_proj = self.norm_aux1(self.aux_proj(a))
        a_refined = self.cross_attn(query=a_proj, value=T, key=T, attention_mask=mask, training=training)
        a_refined = self.norm_aux2(a_proj + self.drop(a_refined, training=training))

        return T, a_refined
