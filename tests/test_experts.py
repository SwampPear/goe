import torch
import pytest

from experts import Experts, ExpertsConfig


def make_experts_and_inputs(
    batch_size: int = 2,
    num_experts: int = 4,
    state_dim: int = 32,
    shared_mlp: bool = True,
):
    cfg = ExpertsConfig(
        num_experts=num_experts,
        state_dim=state_dim,
        hidden_dim=64,
        num_mlp_layers=2,
        message_passing_steps=2,
        dropout=0.1,
        shared_mlp=shared_mlp,
        use_layer_norm=True,
    )
    experts = Experts(cfg)

    expert_states = torch.randn(batch_size, num_experts, state_dim)

    # Random adjacency, row-normalized
    adj_logits = torch.randn(num_experts, num_experts)
    adjacency = torch.softmax(adj_logits, dim=-1)

    return experts, expert_states, adjacency, cfg


def test_experts_forward_shared_mlp_shapes_and_gradients():
    experts, expert_states, adjacency, cfg = make_experts_and_inputs(
        shared_mlp=True
    )

    expert_states.requires_grad_(True)

    out = experts(expert_states, adjacency)

    B, M, D = expert_states.shape
    assert out.shape == (B, M, D)

    # Check gradients flow
    loss = out.mean()
    loss.backward()

    assert expert_states.grad is not None
    assert torch.isfinite(expert_states.grad).all()

    # Parameters get gradients
    has_param_grad = any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in experts.parameters()
    )
    assert has_param_grad


def test_experts_forward_per_expert_mlp():
    experts, expert_states, adjacency, cfg = make_experts_and_inputs(
        shared_mlp=False
    )

    out = experts(expert_states, adjacency)

    B, M, D = expert_states.shape
    assert out.shape == (B, M, D)


def test_experts_raises_on_bad_expert_count():
    experts, expert_states, adjacency, cfg = make_experts_and_inputs()

    # Change M dimension
    bad_states = torch.randn(expert_states.shape[0], cfg.num_experts + 1, cfg.state_dim)

    with pytest.raises(ValueError):
        _ = experts(bad_states, adjacency)


def test_experts_raises_on_bad_adjacency_shape():
    experts, expert_states, adjacency, cfg = make_experts_and_inputs()

    # Wrong adjacency shape
    bad_adj = torch.randn(cfg.num_experts + 1, cfg.num_experts + 1)

    with pytest.raises(ValueError):
        _ = experts(expert_states, bad_adj)
