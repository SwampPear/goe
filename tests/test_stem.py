# test_stem.py
import torch
import pytest

from stem import InputStem, InputStemConfig


def make_stem_and_input(
    batch_size: int = 2,
    in_channels: int = 1,
    depth: int = 32,
    height: int = 64,
    width: int = 64,
):
    cfg = InputStemConfig(
        in_channels=in_channels,
        stem_channels=16,
        token_dim=32,
        aux_dim=8,
        patch_size=(4, 4, 4),
        patch_stride=(2, 2, 2),
        use_layer_norm=True,
    )
    stem = InputStem(cfg)
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return stem, x, cfg


def test_stem_forward_shapes():
    stem, x, cfg = make_stem_and_input()
    tokens, aux, meta = stem(x)

    B = x.shape[0]
    assert tokens.dim() == 3
    assert aux.dim() == 3

    # Check batch dimension and embedding sizes
    assert tokens.shape[0] == B
    assert aux.shape[0] == B
    assert tokens.shape[-1] == cfg.token_dim
    assert aux.shape[-1] == cfg.aux_dim

    # N should match grid_size product
    grid_size = meta["grid_size"]
    assert grid_size.numel() == 3
    expected_N = int(grid_size[0] * grid_size[1] * grid_size[2])
    assert tokens.shape[1] == expected_N
    assert aux.shape[1] == expected_N

    # coords should match N and be in [0, 1]
    coords = meta["coords"]
    assert coords.shape == (B, expected_N, 3)
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= 1.0)


def test_stem_lazy_init_is_stable_on_second_call():
    stem, x, _ = make_stem_and_input()
    # First call triggers lazy init
    t1, a1, m1 = stem(x)
    # Second call should reuse projections without error
    t2, a2, m2 = stem(x)

    assert t1.shape == t2.shape
    assert a1.shape == a2.shape
    assert m1["grid_size"].tolist() == m2["grid_size"].tolist()


def test_stem_gradients_flow():
    stem, x, _ = make_stem_and_input()
    x.requires_grad_(True)

    tokens, aux, _ = stem(x)
    loss = tokens.mean() + aux.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_invalid_input_shape_raises():
    stem, x, _ = make_stem_and_input()
    # Drop the depth dimension to break the [B, C, D, H, W] assumption
    bad_x = torch.randn(x.shape[0], x.shape[1], x.shape[3], x.shape[4])

    with pytest.raises(ValueError):
        _ = stem(bad_x)


def test_patch_config_produces_tokens():
    # Use small spatial dims with compatible patch/stride to make sure we still
    # get at least one patch.
    stem, x, cfg = make_stem_and_input(depth=16, height=16, width=16)
    tokens, aux, meta = stem(x)

    B = x.shape[0]
    N = tokens.shape[1]

    assert N > 0
    assert tokens.shape == (B, N, cfg.token_dim)
    assert aux.shape == (B, N, cfg.aux_dim)
    assert meta["coords"].shape == (B, N, 3)
