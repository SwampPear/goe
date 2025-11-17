import torch
import pytest

from router import GraphRouter, GraphRouterConfig


def make_router_and_inputs(
    batch_size: int = 2,
    num_tokens: int = 64,
    token_dim: int = 128,
    aux_dim: int = 32,
    num_experts: int = 4,
):
    cfg = GraphRouterConfig(
        token_dim=token_dim,
        aux_dim=aux_dim,
        num_experts=num_experts,
        hidden_dim=64,
        expert_state_dim=64,
        temperature=1.0,
        lambda_balance=1e-2,
        lambda_entropy=1e-3,
        use_aux=True,
    )
    router = GraphRouter(cfg)

    tokens = torch.randn(batch_size, num_tokens, token_dim)
    aux = torch.randn(batch_size, num_tokens, aux_dim)

    key_padding_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    # Mask second half of tokens in the second sequence
    key_padding_mask[1, num_tokens // 2 :] = True

    return router, tokens, aux, key_padding_mask, cfg


def test_router_forward_shapes_and_losses():
    router, tokens, aux, key_padding_mask, cfg = make_router_and_inputs()
    out = router(tokens, aux, key_padding_mask=key_padding_mask)

    B, N, _ = tokens.shape
    M = cfg.num_experts
    Dh = router.expert_state_dim

    logits = out["logits"]
    probs = out["probs"]
    expert_states = out["expert_states"]
    adjacency = out["adjacency"]
    load_balance_loss = out["load_balance_loss"]
    entropy_loss = out["entropy_loss"]

    # Basic shapes
    assert logits.shape == (B, N, M)
    assert probs.shape == (B, N, M)
    assert expert_states.shape == (B, M, Dh)
    assert adjacency.shape == (M, M)

    # Row-normalized adjacency
    row_sums = adjacency.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # Losses are scalars and finite
    for loss in (load_balance_loss, entropy_loss):
        assert loss.dim() == 0
        assert torch.isfinite(loss)


def test_router_gradients_flow():
    router, tokens, aux, key_padding_mask, _ = make_router_and_inputs()
    tokens.requires_grad_(True)
    aux.requires_grad_(True)

    out = router(tokens, aux, key_padding_mask=key_padding_mask)
    loss = out["expert_states"].mean() + out["load_balance_loss"] + out["entropy_loss"]
    loss.backward()

    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert aux.grad is not None
    assert torch.isfinite(aux.grad).all()

    # Router parameters should also get gradients
    has_param_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in router.parameters()
    )
    assert has_param_grad


def test_router_respects_key_padding_mask():
    router, tokens, aux, key_padding_mask, cfg = make_router_and_inputs(
        batch_size=2,
        num_tokens=8,
    )
    out = router(tokens, aux, key_padding_mask=key_padding_mask)
    probs = out["probs"]  # [B, N, M]

    # All probs at masked positions should be ~0 after masking + softmax
    masked_probs = probs[1, 4:, :]  # second example, positions 4..7 are masked
    assert torch.allclose(masked_probs, torch.zeros_like(masked_probs), atol=1e-6)


def test_router_raises_on_bad_token_shape():
    router, tokens, aux, key_padding_mask, _ = make_router_and_inputs()
    bad_tokens = torch.randn(tokens.shape[0], tokens.shape[1], tokens.shape[2] + 1)

    with pytest.raises(ValueError):
        _ = router(bad_tokens, aux, key_padding_mask=key_padding_mask)


def test_router_raises_when_aux_missing_but_required():
    router, tokens, aux, key_padding_mask, _ = make_router_and_inputs()

    with pytest.raises(ValueError):
        _ = router(tokens, aux=None, key_padding_mask=key_padding_mask)
