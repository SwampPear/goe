import tensorflow as tf
from tensorflow.keras import layers

class Recombination(layers.Layer):
    """
    t_next[i] = Σ_m p[i,m] * W_o h[m]
    Inputs:
      p:   [B, N, M]   routing probs
      H:   [B, M, d_h] expert states
    Returns:
      Tn:  [B, N, d_out] token embeddings
    """
    def __init__(self, d_h: int, d_out: int, name='combine_to_tokens'):
        super().__init__(name=name)
        self.Wo = layers.Dense(d_out)  # per-expert projection

    def call(self, p: tf.Tensor, H: tf.Tensor):
        H_proj = self.Wo(H)                 # [B, M, d_out]
        T_next = tf.einsum('bnm,bmd->bnd', p, H_proj)  # weighted sum over experts
        return T_next
