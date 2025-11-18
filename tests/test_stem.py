import math

import pytest
import torch

from src.models.stem import (
    InputStemConfig,
    Conv3DEncoder,
    InputStem,
    extract_patches_3d,
)


def test_input_stem_config_defaults():
    cfg = InputStemConfig()  # use dataclass defaults
    assert isinstance(cfg.stem_channels, int) and cfg.stem_channels > 0
    assert isinstance(cfg.token_dim, int) and cfg.token_dim > 0
    assert isinstance(cfg.aux_dim, int) and cfg.aux_dim > 0
    assert isinstance(cfg.patch_size, tuple) and len(cfg.patch_size) == 3
    assert isinstance(cfg.patch_stride, tuple) and len(cfg.patch_stride) == 3
    assert isinstance(cfg.use_layer_norm, bool)


def test_conv3d_encoder_output_shape():
    B, D, H, W = 2, 32, 48, 64
    stem_channels = 24

    encoder = Conv3DEncoder(stem_channels=stem_channels)
    x = torch.randn(B, 1, D, H, W)

    y = encoder(x)

    # basic structural checks
    assert y.ndim == 5
    assert y.shape[0] == B
    assert y.shape[1] == stem_channels

    # spatial dims should be positive and not exceed input dims
    Dy, Hy, Wy = y.shape[2:]
    assert Dy > 0 and Hy > 0 and Wy > 0
    assert Dy <= D and Hy <= H and Wy <= W


def test_extract_patches_3d_shapes_and_grid():
    B, C, D, H, W = 2, 4, 10, 12, 14
    features = torch.randn(B, C, D, H, W)

    patch_size = (3, 4, 5)
    stride = (2, 3, 4)

    patches, coords = extract_patches_3d(features, patch_size, stride)

    kD, kH, kW = patch_size
    sD, sH, sW = stride

    D_out = math.floor((D - kD) / sD) + 1
    H_out = math.floor((H - kH) / sH) + 1
    W_out = math.floor((W - kW) / sW) + 1
    N = D_out * H_out * W_out

    # patch tensor shape
    assert patches.shape == (B, N, C * kD * kH * kW)
    # coordinate tensor shape
    assert coords.shape == (B, N, 3)


def test_input_stem_lazy_init_and_output_shapes():
    # choose small but valid patch params
    cfg = InputStemConfig(
        stem_channels=16,
        token_dim=32,
        aux_dim=8,
        patch_size=(4, 4, 4),
        patch_stride=(2, 2, 2),
        use_layer_norm=True,
    )
    stem = InputStem(cfg)

    assert stem._initialized is False
    assert stem.token_proj is None
    assert stem.aux_proj is None

    B, D, H, W = 2, 16, 16, 16
    x = torch.randn(B, 1, D, H, W)

    tokens, aux, meta = stem(x)

    # lazy init should have run
    assert stem._initialized is True
    assert stem.token_proj is not None
    assert stem.aux_proj is not None

    # shapes
    assert tokens.ndim == 3
    assert aux.ndim == 3
    assert tokens.shape[0] == B
    assert aux.shape[0] == B
    assert tokens.shape[2] == cfg.token_dim
    assert aux.shape[2] == cfg.aux_dim
    assert tokens.shape[1] == aux.shape[1]  # same number of tokens N

    # metadata checks
    assert set(meta.keys()) == {"coords", "grid_size", "patch_size", "volume_shape"}
    assert meta["coords"].shape[0] == B
    assert meta["coords"].shape[1] == tokens.shape[1]
    assert meta["coords"].shape[2] == 3
    assert meta["grid_size"].shape == (3,)
    assert meta["patch_size"].shape == (3,)
    assert meta["volume_shape"].shape == (3,)

    # LayerNorm branch
    assert isinstance(stem.token_norm, torch.nn.LayerNorm)


def test_input_stem_without_layer_norm():
    cfg = InputStemConfig(
        stem_channels=8,
        token_dim=16,
        aux_dim=4,
        patch_size=(2, 2, 2),
        patch_stride=(2, 2, 2),
        use_layer_norm=False,
    )
    stem = InputStem(cfg)

    x = torch.randn(1, 1, 8, 8, 8)
    tokens, aux, _ = stem(x)

    assert tokens.shape[0] == 1
    assert aux.shape[0] == 1
    assert tokens.shape[2] == cfg.token_dim
    assert aux.shape[2] == cfg.aux_dim
    # Identity branch when use_layer_norm=False
    from torch.nn import Identity

    assert isinstance(stem.token_norm, Identity)


def test_input_stem_invalid_patch_config_raises():
    # absurdly large patch size so that no patches fit -> ValueError
    cfg = InputStemConfig(
        stem_channels=8,
        token_dim=16,
        aux_dim=4,
        patch_size=(999, 999, 999),
        patch_stride=(1, 1, 1),
        use_layer_norm=True,
    )
    stem = InputStem(cfg)

    x = torch.randn(1, 1, 16, 16, 16)

    with pytest.raises(ValueError):
        _ = stem(x)


def test_input_stem_reuse_does_not_reinitialize_linears():
    cfg = InputStemConfig(
        stem_channels=16,
        token_dim=32,
        aux_dim=8,
        patch_size=(4, 4, 4),
        patch_stride=(2, 2, 2),
        use_layer_norm=True,
    )
    stem = InputStem(cfg)
    x = torch.randn(1, 1, 16, 16, 16)

    # first call triggers lazy init
    tokens1, aux1, _ = stem(x)
    token_proj_id = id(stem.token_proj)
    aux_proj_id = id(stem.aux_proj)

    # second call should reuse same linear layers
    tokens2, aux2, _ = stem(x)

    assert id(stem.token_proj) == token_proj_id
    assert id(stem.aux_proj) == aux_proj_id
    assert tokens1.shape == tokens2.shape
    assert aux1.shape == aux2.shape
