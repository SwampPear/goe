import tensorflow as tf
from tensorflow.keras import layers


class Router(layers.Layer):
    """
    Graph Router (Sec. 3.1.3)

    r_i = concat(t'_i, a'_i)
    p_i = softmax((W_r r_i)/τ)          # optional temperature τ
    h_in[m] = Σ_i p_{i,m} * W_p r_i
    A = row_softmax(A_logits + I)        # static row-stochastic adjacency (bias self-loops)

    Inputs:
      T_prime:      [B, N, d_token]
      a_prime:      [B, N, d_aux]
      expert_mask:  optional [M] or [B, N, M] (1=allowed, 0=blocked)
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
        self.gate = layers.Dense(self.M, use_bias=True)   # W_r
        self.r_proj = layers.Dense(d_exp)                 # W_p
        self.A_logits = self.add_weight(
            name='A_logits', shape=(self.M, self.M),
            initializer='zeros', trainable=True
        )

    def build(self, input_shape):
        """Keras build step."""
        # identity bias for self-loops; match compute dtype to avoid AMP issues
        self._eye = tf.eye(self.M, dtype=self.variable_dtype)
        super().build(input_shape)

    def call(self, T_prime, a_prime, training=False):
        """
        Steps:
            (1) r      = concat(T', a')                                 # token+aux descriptor
            (2) logits = W_r r                                          # gate logits
                if expert_mask: broadcast to [B, N, M]; mask==0 -> -inf
            (3) p      = softmax(logits / τ) → Dropout(p) if training   # routing probs
            (4) r_proj = W_p r                                          # expert input projection
            (5) h_in   = einsum('bnm,bnd->bmd', p, r_proj)              # token→expert aggregation
            (6) A      = row_softmax(A_logits + I)                      # row-stochastic adjacency

        Args:
            T_prime: contextualized tokens [B, N, d_token]
            a_prime: refined auxiliary features [B, N, d_aux]
            expert_mask: optional mask over experts; shape [M] or [B, N, M]
                        (1=allowed, 0=blocked). If [M], it is broadcast to [1,1,M].
            training: enables dropout on routing probabilities when True

        Returns:
            p:    routing probabilities [B, N, M]
            A:    static row-stochastic adjacency [M, M]
            h_in: per-expert aggregated inputs [B, M, d_exp]
            r:    concatenated descriptors [B, N, d_token + d_aux]
        """
        # r = [t' ; a']
        r = self.concat([T_prime, a_prime])               # [B, N, d_r]
        p = tf.nn.softmax(self.gate(r), axis=-1)          # [B, N, M]
        h_in = tf.einsum('bnm,bnd->bmd', p, self.r_proj(r))
        A = tf.nn.softmax(self.A_logits + self._eye, axis=-1)
        return p, A, h_in
