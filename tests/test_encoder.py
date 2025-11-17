import torch
import pytest

from encoder import Encoder, EncoderConfig


def make_encoder_and_inputs(
    batch_size: int = 2,
    num_tokens: int = 64,
    token_dim: int = 128,
    aux_dim: int = 32,
):
    cfg = EncoderConfig(
        token_dim=token_dim,
        aux_dim=aux_dim,
        num_layers=2,
        num_heads=8,
        ff_hidden_dim=256,
        dropout=0.1,
        use_positional_encoding=True,
        use_aux_update=True,
    )
    enc = Encoder(cfg)

    tokens = torch.randn(batch_size, num_tokens, token_dim)
    aux = torch.randn(batch_size, num_tokens, aux_dim)

    # Mask half of the tokens in the second sequence as padding
    key_padding_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    key_padding_mask[1, num_tokens // 2 :] = True

    return enc, tokens, aux, key_padding_mask, cfg


def test_encoder_forward_shapes():
    enc, tokens, aux, key_padding_mask, cfg = make_encoder_and_inputs()
    out_tokens, out_aux = enc(tokens, aux, key_padding_mask=key_padding_mask)

    B, N, _ = tokens.shape
    assert out_tokens.shape == (B, N, cfg.token_dim)
    assert out_aux is not None
    assert out_aux.shape == (B, N, cfg.aux_dim)


def test_encoder_no_aux_still_works():
    enc, tokens, aux, key_padding_mask, cfg = make_encoder_and_inputs()
    out_tokens, out_aux = enc(tokens, aux=None, key_padding_mask=key_padding_mask)

    B, N, _ = tokens.shape
    assert out_tokens.shape == (B, N, cfg.token_dim)
    assert out_aux is None


def test_encoder_gradients_flow():
    enc, tokens, aux, key_padding_mask, _ = make_encoder_and_inputs()
    tokens.requires_grad_(True)
    aux.requires_grad_(True)

    out_tokens, out_aux = enc(tokens, aux, key_padding_mask=key_padding_mask)
    loss = out_tokens.mean() + out_aux.mean()
    loss.backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert aux.grad is not None
    assert torch.isfinite(aux.grad).all()


def test_encoder_raises_on_bad_token_dim():
    enc, tokens, aux, key_padding_mask, cfg = make_encoder_and_inputs()
    bad_tokens = torch.randn(tokens.shape[0], tokens.shape[1], cfg.token_dim + 1)

    with pytest.raises(ValueError):
        _ = enc(bad_tokens, aux, key_padding_mask=key_padding_mask)
