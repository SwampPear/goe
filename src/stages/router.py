import tensorflow as tf
from tensorflow.keras import layers

class Router(layers.Layer):
    """
    Minimal Graph Router (Sec. 3.1.3)
      r_i = concat(t'_i, a'_i)
      p_i = softmax(W_r r_i)
      h_in[m] = Σ_i p_{i,m} * W_p r_i
      A: static learnable row-stochastic adjacency
    Inputs:
      T_prime: [B, N, d_token]
      a_prime: [B, N, d_aux]
    Outputs:
      p:    [B, N, M]
      A:    [M, M]
      h_in: [B, M, d_exp]
      r:    [B, N, d_token + d_aux]
    """
    def __init__(self, num_experts: int, d_token: int, d_aux: int, d_exp: int):
        super().__init__()
        self.M = int(num_experts)
        self.concat = layers.Concatenate(axis=-1)
        self.gate = layers.Dense(self.M)         # W_r
        self.r_proj = layers.Dense(d_exp)        # W_p
        self.A_logits = self.add_weight(         # static adjacency logits
            name='A_logits', shape=(self.M, self.M),
            initializer='zeros', trainable=True
        )

    def build(self, input_shape):
        self._eye = tf.eye(self.M, dtype=self.dtype)  # identity bias for self-loops
        super().build(input_shape)

    def call(self, T_prime: tf.Tensor, a_prime: tf.Tensor):
        # r_i = [t'_i ; a'_i]
        r = self.concat([T_prime, a_prime])           # [B, N, d_r]

        # p_i = softmax(W_r r_i)
        logits = self.gate(r)                         # [B, N, M]
        p = tf.nn.softmax(logits, axis=-1)            # [B, N, M]

        # h_in[m] = Σ_i p_{i,m} * W_p r_i
        r_proj = self.r_proj(r)                       # [B, N, d_exp]
        h_in = tf.einsum('bnm,bnd->bmd', p, r_proj)   # [B, M, d_exp]

        # Static row-stochastic adjacency A
        A = tf.nn.softmax(self.A_logits + self._eye, axis=-1)  # [M, M]

        return p, A, h_in, r
