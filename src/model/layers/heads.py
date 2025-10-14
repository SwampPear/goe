import tensorflow as tf
from tensorflow.keras import layers
from typing import Optional, Tuple, Dict, Any, List

class _ProjectAndNorm(layers.Layer):
    """
    Lightweight projection + normalization used inside heads.

    Args:
        d_out: projection width
        dropout: dropout rate applied after projection
        use_ln: if True, apply LayerNorm before projection (pre-norm)
    """
    def __init__(self, d_out: int, dropout: float = 0.0, use_ln: bool = True):
        super().__init__()
        self.use_ln = use_ln
        self.ln = layers.LayerNormalization(epsilon=1e-6) if use_ln else None
        self.proj = layers.Dense(d_out)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=False):
        if self.use_ln:
            x = self.ln(x)
        x = self.proj(x)
        return self.drop(x, training=training)


class TokenClassificationHead(layers.Layer):
    """
    Per-token classification head (e.g., sequence tagging, patch/voxel tagging).

    Inputs:
        h: Tensor of shape [B, N, d] token embeddings.
        a: Optional Tensor of shape [B, N, da] auxiliary features aligned with tokens.

    Args:
        num_classes: number of classes to predict per token
        hidden_dim: MLP hidden width before logits
        dropout: dropout rate inside the MLP
        use_aux: if True, concatenate auxiliary features 'a' to 'h'
        name: optional layer name

    Call returns:
        logits: [B, N, num_classes]
    """
    def __init__(self,
                 num_classes: int,
                 hidden_dim: int = 0,
                 dropout: float = 0.0,
                 use_aux: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.use_aux = use_aux
        self.hidden = _ProjectAndNorm(hidden_dim, dropout) if hidden_dim > 0 else None
        self.out = layers.Dense(num_classes)

    def call(self, h: tf.Tensor, a: Optional[tf.Tensor] = None, training=False) -> tf.Tensor:
        x = h if (not self.use_aux or a is None) else tf.concat([h, a], axis=-1)
        if self.hidden is not None:
            x = self.hidden(x, training=training)
            x = tf.nn.gelu(x)
        return self.out(x)


class DenseSegmentationHead(layers.Layer):
    """
    Dense segmentation head that maps token grid back to spatial logits.
    Use when your tokens are a uniform HxW (or D x H x W) grid.

    Inputs:
        h: [B, N, d] token embeddings (N = Π grid dims)
        grid_shape: tuple of ints describing the spatial layout (e.g., (H, W) or (D, H, W))

    Args:
        num_classes: number of output channels/classes
        grid_shape: spatial shape used to fold tokens back (must multiply to N)
        hidden_dim: projection width before the final conv
        dropout: dropout rate inside the projection
        conv_kernel: kernel size for final refinement conv (set 1 for linear)
        use_aux: if True, concatenate auxiliary features to token embeddings before folding
        name: optional layer name

    Call returns:
        logits: [B, *grid_shape, num_classes]
    """
    def __init__(self,
                 num_classes: int,
                 grid_shape: Tuple[int, ...],
                 hidden_dim: int = 0,
                 dropout: float = 0.0,
                 conv_kernel: int = 1,
                 use_aux: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.grid_shape = tuple(grid_shape)
        self.use_aux = use_aux
        self.hidden = _ProjectAndNorm(hidden_dim, dropout) if hidden_dim > 0 else None
        self.conv_kernel = conv_kernel
        self.final_conv = None  # built at call time for correct rank

        # lightweight reshape helpers
        self._rank = len(self.grid_shape)

    def _fold(self, x: tf.Tensor) -> tf.Tensor:
        # [B, N, C] -> [B, *grid, C]
        b = tf.shape(x)[0]
        c = x.shape[-1]
        return tf.reshape(x, tf.concat([[b], tf.constant(self.grid_shape, dtype=tf.int32), [c]], axis=0))

    def call(self, h: tf.Tensor, a: Optional[tf.Tensor] = None, training=False) -> tf.Tensor:
        x = h if (not self.use_aux or a is None) else tf.concat([h, a], axis=-1)
        if self.hidden is not None:
            x = self.hidden(x, training=training)
            x = tf.nn.gelu(x)

        x = self._fold(x)  # [B, *grid, C]

        # Build a rank-appropriate ConvNd lazily
        if self.final_conv is None:
            if self._rank == 1:
                self.final_conv = layers.Conv1D(filters=self.num_classes, kernel_size=self.conv_kernel, padding="same")
            elif self._rank == 2:
                self.final_conv = layers.Conv2D(filters=self.num_classes, kernel_size=self.conv_kernel, padding="same")
            elif self._rank == 3:
                self.final_conv = layers.Conv3D(filters=self.num_classes, kernel_size=self.conv_kernel, padding="same")
            else:
                raise ValueError(f"Unsupported grid rank: {self._rank}")

        return self.final_conv(x)

    @property
    def num_classes(self):
        # infer from built conv if present
        if self.final_conv is not None:
            return self.final_conv.filters
        return None  # only used before build


class RegressionHead(layers.Layer):
    """
    Token or sequence regression head.

    Modes:
        - "token": per-token regression -> output [B, N, out_dim]
        - "pooled": sequence regression via pooling -> output [B, out_dim]

    Inputs:
        h: [B, N, d] token embeddings
        a: optional [B, N, da] aux features

    Args:
        out_dim: number of regression targets
        mode: "token" or "pooled"
        pool: pooling op when mode="pooled": "mean" | "max" | "cls" (uses h[:,0])
        hidden_dim: MLP hidden width
        dropout: dropout inside MLP
        use_aux: if True, concatenate aux features to tokens

    Returns:
        y: regression tensor with shape per mode
    """
    def __init__(self,
                 out_dim: int,
                 mode: str = "pooled",
                 pool: str = "mean",
                 hidden_dim: int = 0,
                 dropout: float = 0.0,
                 use_aux: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name)
        assert mode in ("token", "pooled")
        assert pool in ("mean", "max", "cls")
        self.out_dim = out_dim
        self.mode = mode
        self.pool = pool
        self.use_aux = use_aux
        self.hidden = _ProjectAndNorm(hidden_dim, dropout) if hidden_dim > 0 else None
        self.out = layers.Dense(out_dim)

    def _pool(self, x: tf.Tensor) -> tf.Tensor:
        if self.pool == "mean":
            return tf.reduce_mean(x, axis=1)
        if self.pool == "max":
            return tf.reduce_max(x, axis=1)
        return x[:, 0]  # "cls" token

    def call(self, h: tf.Tensor, a: Optional[tf.Tensor] = None, training=False) -> tf.Tensor:
        x = h if (not self.use_aux or a is None) else tf.concat([h, a], axis=-1)
        if self.hidden is not None:
            x = self.hidden(x, training=training)
            x = tf.nn.gelu(x)
        if self.mode == "pooled":
            x = self._pool(x)
        return self.out(x)


# --------- Factory & training helpers ----------

def make_head(task: str, *, cfg: Dict[str, Any]) -> layers.Layer:
    """
    Factory for building a head.

    Args:
        task: one of {"token_cls", "segmentation", "regression"}
        cfg: dict of kwargs for the specific head

    Returns:
        A Keras Layer implementing the requested head.
    """
    if task == "token_cls":
        return TokenClassificationHead(**cfg)
    if task == "segmentation":
        return DenseSegmentationHead(**cfg)
    if task == "regression":
        return RegressionHead(**cfg)
    raise ValueError(f"Unknown task '{task}'")


def default_loss_and_metrics(task: str,
                             num_classes: Optional[int] = None) -> Tuple[tf.keras.losses.Loss, List[tf.keras.metrics.Metric]]:
    """
    Sensible defaults per task for training.

    token_cls / segmentation: SparseCategoricalCrossentropy (from_logits=True)
    regression: MeanSquaredError

    Returns:
        (loss, metrics)
    """
    if task in ("token_cls", "segmentation"):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        if num_classes is not None and num_classes <= 2:
            # optional extra: binary quality when it's effectively binary
            metrics.append(tf.keras.metrics.AUC(name="auc"))
        return loss, metrics
    if task == "regression":
        return tf.keras.losses.MeanSquaredError(), [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    raise ValueError(f"Unknown task '{task}'")
