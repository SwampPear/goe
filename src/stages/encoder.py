import tensorflow as tf
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.attn(x, x)
        x = self.norm1(x + self.drop(attn_output, training=training))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_output, training=training))
        
        return x


class ModalityAgnosticEncoder(layers.Layer):
    def __init__(self, d_model=256, num_heads=8, ff_dim=512, num_layers=4, d_aux=64, dropout=0.1):
        super().__init__()
        self.encoder_layers = [
            TransformerBlock(d_model, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ]
        self.aux_proj = layers.Dense(d_aux)
        self.cross_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm_aux = layers.LayerNormalization(epsilon=1e-6)

    def call(self, T, a, training=False):
        # T: [B, N, d_model] tokens
        # a: [B, N, d_aux]   auxiliary routing features
        for block in self.encoder_layers:
            T = block(T, training=training)       # Φ(T; θ_e)
        # faux(a, T′): refine auxiliary routing features with contextual tokens
        a_proj = self.aux_proj(a)
        a_refined = self.cross_attn(query=a_proj, value=T, key=T)
        a_refined = self.norm_aux(a_proj + a_refined)
        return T, a_refined                      # (T′, a′)
