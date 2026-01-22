import torch

from src.models.stem import InputStemConfig
from src.models.encoder import EncoderConfig
from src.models.router import GraphRouterConfig
from src.models.experts import ExpertsConfig
from src.models.decoder import DecoderConfig
from src.models.goe import GraphOfExperts, GraphOfExpertsConfig


def _build_small_model() -> GraphOfExperts:
    stem_cfg = InputStemConfig(
        stem_channels=8,
        token_dim=16,
        aux_dim=4,
        patch_size=(2, 2, 2),
        patch_stride=(2, 2, 2),
        use_layer_norm=True,
    )
    encoder_cfg = EncoderConfig(
        token_dim=16,
        aux_dim=4,
        num_layers=1,
        num_heads=4,
        ff_hidden_dim=32,
        dropout=0.0,
        use_positional_encoding=False,
        use_aux_update=True,
    )
    router_cfg = GraphRouterConfig(
        token_dim=16,
        aux_dim=4,
        num_experts=3,
        hidden_dim=16,
        expert_state_dim=16,
        temperature=1.0,
        lambda_balance=1e-2,
        lambda_entropy=1e-3,
        use_aux=True,
    )
    experts_cfg = ExpertsConfig(
        num_experts=3,
        state_dim=16,
        hidden_dim=32,
        num_mlp_layers=2,
        message_passing_steps=1,
        dropout=0.0,
        shared_mlp=True,
        use_layer_norm=False,
    )
    decoder_cfg = DecoderConfig(
        token_dim=16,
        expert_state_dim=16,
        num_experts=3,
        out_channels=2,
        hidden_dim=16,
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
    return GraphOfExperts(cfg)


def test_forward_shapes_and_probs():
    torch.manual_seed(0)
    model = _build_small_model()
    x = torch.randn(2, 1, 16, 16, 16)

    out = model(x, return_intermediates=True)
    assert "logits" in out
    assert "routing" in out

    logits = out["logits"]
    assert logits.shape == (2, 2, 4, 4, 4)

    probs = out["routing"]["probs"]
    assert probs.shape[0] == 2
    assert probs.shape[-1] == 3

    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    assert torch.isfinite(out["aux_loss"]).all()
