import tensorflow as tf
from tensorflow.keras import layers
from typing import List

class ExpertMLP(layers.Layer):
    """Single expert f_m: modality-agnostic MLP."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.0, name: str | None = None):
        super().__init__(name=name)
        self.net = tf.keras.Sequential([
            layers.Dense(d_hidden, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(d_out),
        ])

    def call(self, x, training=False):
        return self.net(x, training=training)  # [B, d_out]


class ExpertsBank(layers.Layer):
    """
    Bank of M experts with independent parameters.

    Inputs:
      h_in: [B, M, d_exp]    per-expert aggregated inputs from the router (Σ_i p_{i,m} W_p r_i)
      A:    [M, M]           static row-stochastic adjacency (from router)

    Outputs:
      H_out: [B, M, d_h]     updated expert states h^{(l+1)}_m
    """
    def __init__(self, num_experts: int, d_exp: int, d_hidden: int, d_h: int, dropout: float = 0.0, name: str = 'experts'):
        super().__init__(name=name)
        self.M = int(num_experts)
        self.d_exp = int(d_exp)
        self.d_hidden = int(d_hidden)
        self.d_h = int(d_h)

        # Create M independent expert modules f_m
        self.experts: List[ExpertMLP] = [
            ExpertMLP(d_in=d_exp, d_hidden=d_hidden, d_out=d_h, dropout=dropout, name=f'expert_{m}')
            for m in range(self.M)
        ]

        # Projection for graph message passing term W_c
        self.W_c = layers.Dense(d_h, name='W_c')

    def call(self, h_in: tf.Tensor, A: tf.Tensor, training=False) -> tf.Tensor:
        """
        h_in: [B, M, d_exp], A: [M, M]
        returns H_out: [B, M, d_h]
        """
        # Local expert updates f_m(h̃_m)
        # Loop over experts to ensure parameter independence
        local_updates = []
        for m in range(self.M):
            hm_in = h_in[:, m, :]                       # [B, d_exp]
            hm_loc = self.experts[m](hm_in, training=training)  # [B, d_h]
            local_updates.append(hm_loc)
        H_loc = tf.stack(local_updates, axis=1)         # [B, M, d_h]

        # Graph message passing: Σ_n A_{m n} * W_c * h_n
        H_proj = self.W_c(H_loc)                        # [B, M, d_h]
        H_ctx  = tf.einsum('mn,bnd->bmd', A, H_proj)    # [B, M, d_h]

        # Combine: local + context
        H_out = H_loc + H_ctx                            # [B, M, d_h]
        return H_out

    def get_config(self):
        return {
            'num_experts': self.M,
            'd_exp': self.d_exp,
            'd_hidden': self.d_hidden,
            'd_h': self.d_h,
            'name': self.name,
        }
