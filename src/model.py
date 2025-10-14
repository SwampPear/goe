import tensorflow as tf
from tensorflow.keras import layers

# ---- 1) Expert-to-Token Readout ----
class CombineToTokens(layers.Layer):
    def __init__(self, d_h: int, d_out: int, name='combine_to_tokens'):
        super().__init__(name=name)
        self.Wo = layers.Dense(d_out)
    def call(self, p: tf.Tensor, H: tf.Tensor) -> tf.Tensor:
        # p: [B,N,M], H: [B,M,d_h] -> T_next: [B,N,d_out]
        H_proj = self.Wo(H)
        return tf.einsum('bnm,bmd->bnd', p, H_proj)

# ---- 2) Simple per-token head (classifier/regressor) ----
class TokenHead(layers.Layer):
    def __init__(self, out_dim: int, activation=None, name='token_head'):
        super().__init__(name=name)
        self.proj = layers.Dense(out_dim, activation=activation)
    def call(self, T_next: tf.Tensor) -> tf.Tensor:  # [B,N,d_in] -> [B,N,out_dim]
        return self.proj(T_next)

# ---- 3) Routing regularizers from §3.1.4 ----
def load_balance_loss(p: tf.Tensor, lambda_b: float) -> tf.Tensor:
    # p: [B,N,M]; encourage mean usage per expert ≈ 1/M
    mean_per_expert = tf.reduce_mean(p, axis=[0,1])        # [M]
    M = tf.cast(tf.shape(p)[-1], p.dtype)
    target = tf.fill(tf.shape(mean_per_expert), 1.0 / M)   # [M]
    return tf.cast(lambda_b, p.dtype) * tf.reduce_sum(tf.square(mean_per_expert - target))

def routing_entropy_loss(p: tf.Tensor, lambda_e: float, mask: tf.Tensor | None = None) -> tf.Tensor:
    # L_entropy = -λ_e Σ_i Σ_m p log p  (use mean over valid tokens for scale)
    eps = tf.constant(1e-8, p.dtype)
    ent = -tf.reduce_sum(p * tf.math.log(tf.clip_by_value(p, eps, 1.0)), axis=-1)  # [B,N]
    if mask is not None:
        mask = tf.cast(mask, ent.dtype)
        ent = tf.reduce_sum(ent * mask) / (tf.reduce_sum(mask) + eps)
    else:
        ent = tf.reduce_mean(ent)
    return tf.cast(lambda_e, p.dtype) * (-ent)  # minus sign per paper spec

# ---- 4) GoE block: router -> experts -> readout ----
class GoEBlock(layers.Layer):
    def __init__(self, router, experts_bank, d_h: int, d_out: int, name='goe_block'):
        super().__init__(name=name)
        self.router = router              # GraphRouter layer (minimal version)
        self.experts = experts_bank       # ExpertsBank layer
        self.readout = CombineToTokens(d_h=d_h, d_out=d_out)
    def call(self, T_prime: tf.Tensor, a_prime: tf.Tensor, training=False):
        # router
        p, A, h_in, r = self.router(T_prime, a_prime)    # p:[B,N,M], A:[M,M], h_in:[B,M,d_exp]
        # experts (with intra-graph msg passing)
        H = self.experts(h_in, A, training=training)     # [B,M,d_h]
        # readout back to tokens
        T_next = self.readout(p, H)                      # [B,N,d_out]
        return T_next, {'p': p, 'A': A, 'H': H, 'r': r}

# ---- 5) Minimal end-to-end wrapper (expects encoder outputs) ----
class GraphOfExpertsModel(tf.keras.Model):
    def __init__(self, encoder, router, experts_bank, head_out_dim: int,
                 lambda_b: float = 0.01, lambda_e: float = 0.01, name='GoE'):
        super().__init__(name=name)
        self.encoder = encoder
        self.router = router
        self.experts = experts_bank
        # assume experts_bank outputs d_h; readout d_out = encoder d_model by default
        self.goe = GoEBlock(router=self.router, experts_bank=self.experts,
                            d_h=self.experts.d_h, d_out=encoder.encoder_layers[0].attn.compute_output_shape((None, None, encoder.encoder_layers[0].attn.key_dim))[-1] if hasattr(encoder.encoder_layers[0].attn,'key_dim') else self.experts.d_h)
        self.head = TokenHead(out_dim=head_out_dim, activation=None)
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e

    def call(self, inputs, training=False, mask: tf.Tensor | None = None):
        # inputs: (T, a) already from stem tokenization
        T, a = inputs
        T_prime, a_prime = self.encoder(T, a, training=training)      # [B,N,d], [B,N,d_aux]
        T_next, aux = self.goe(T_prime, a_prime, training=training)   # [B,N,d_out]
        logits = self.head(T_next)                                    # [B,N,C] (or reg dim)
        if training:
            lb = load_balance_loss(aux['p'], self.lambda_b)
            ent = routing_entropy_loss(aux['p'], self.lambda_e, mask=mask)
            self.add_loss(lb + ent)
        return logits
