import torch

from goe import GraphOfExperts, GraphOfExpertsConfig
from stem import InputStemConfig
from encoder import EncoderConfig
from router import GraphRouterConfig
from experts import ExpertsConfig
from decoder import DecoderConfig


def make_small_goe_model():
    # Tiny dimensions so the test runs fast and avoids OOM
    token_dim = 16
    aux_dim = 8
    num_experts = 3
    expert_state_dim = 16
    out_channels = 1

    stem_cfg = InputStemConfig(
        in_channels=1,
        stem_channels=8,
        token_dim=token_dim,
        aux_dim=aux_dim,
        patch_size=(4, 4, 4),
        patch_stride=(4, 4, 4),
        use_layer_norm=True,
    )

    encoder_cfg = EncoderConfig(
        token_dim=token_dim,
        aux_dim=aux_dim,
        num_layers=1,
        num_heads=4,          # 16 / 4 = 4 per head
        ff_hidden_dim=32,
        dropout=0.0,
        use_positional_encoding=False,
        use_aux_update=True,
    )

    router_cfg = GraphRouterConfig(
        token_dim=token_dim,
        aux_dim=aux_dim,
        num_experts=num_experts,
        hidden_dim=32,
        expert_state_dim=expert_state_dim,
        temperature=1.0,
        lambda_balance=1e-2,
        lambda_entropy=1e-3,
        use_aux=True,
    )

    experts_cfg = ExpertsConfig(
        num_experts=num_experts,
        state_dim=expert_state_dim,
        hidden_dim=32,
        num_mlp_layers=2,
        message_passing_steps=1,
        dropout=0.0,
        shared_mlp=True,
        use_layer_norm=True,
    )

    decoder_cfg = DecoderConfig(
        token_dim=token_dim,
        expert_state_dim=expert_state_dim,
        num_experts=num_experts,
        out_channels=out_channels,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        use_coords=True,
    )

    cfg = GraphOfExpertsConfig(
        stem=stem_cfg,
        encoder=encoder_cfg,
        router=router_cfg,
        experts=experts_cfg,
        decoder=decoder_cfg,
    )

    model = GraphOfExperts(cfg)
    return model, cfg


def test_goe_end_to_end_forward_and_backward():
    model, cfg = make_small_goe_model()

    # Small 3D volume: after stem encoder's stride-2, shape will shrink,
    # and patch_size/stride are chosen so some tokens exist.
    batch_size = 1
    x = torch.randn(batch_size, cfg.stem.in_channels, 16, 16, 16, requires_grad=True)

    out = model(x, return_intermediates=True)

    # Check logits exist and have 5D shape [B, C, D, H, W]
    logits = out["logits"]
    assert logits.dim() == 5
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == cfg.decoder.out_channels

    # Aux losses are finite scalars
    aux_loss = out["aux_loss"]
    lb_loss = out["lb_loss"]
    entropy_loss = out["entropy_loss"]

    for loss in (aux_loss, lb_loss, entropy_loss):
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    # Backward through a simple total loss
    total_loss = logits.mean() + aux_loss
    total_loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
