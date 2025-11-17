from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import nn, Tensor

from stem import InputStem, InputStemConfig
from encoder import Encoder, EncoderConfig
from router import GraphRouter, GraphRouterConfig
from experts import Experts, ExpertsConfig
from decoder import Decoder, DecoderConfig


@dataclass
class GraphOfExpertsConfig:
    """
    Top-level configuration for the Graph-of-Experts model.

    This bundles the sub-configs for:
        - Input stem
        - Encoder
        - Graph router
        - Expert refinement
        - Decoder

    Attributes
    ----------
    stem:
        Configuration for the InputStem (φ + patching).
    encoder:
        Configuration for the token encoder Φ(T; θ_e).
    router:
        Configuration for the graph router (routing + expert states).
    experts:
        Configuration for the expert refinement module.
    decoder:
        Configuration for the decoder that maps back to volumetric outputs.
    """

    stem: InputStemConfig
    encoder: EncoderConfig
    router: GraphRouterConfig
    experts: ExpertsConfig
    decoder: DecoderConfig


class GraphOfExperts(nn.Module):
    """
    Full Graph-of-Experts model.

    Pipeline:
        x (volume) → Stem:
            T_0, a_0, meta      = stem(x)

        → Encoder:
            T_enc, a_enc        = encoder(T_0, a_0)

        → Router:
            routing             = router(T_enc, a_enc)
            p                   = routing["probs"]         # token → expert
            h_router            = routing["expert_states"] # initial expert states

        → Experts:
            h_refined           = experts(h_router, routing["adjacency"])

        → Decoder:
            logits              = decoder(T_enc, h_refined, routing_probs=p, meta=meta)

    The task-specific loss (e.g., BCEWithLogits over ink labels) is expected
    to be computed outside this class. This module exposes the logits and
    routing regularizers needed for training.
    """

    def __init__(self, cfg: GraphOfExpertsConfig):
        super().__init__()
        self.cfg = cfg

        # Make sure the shared dimensions line up
        if cfg.encoder.token_dim != cfg.stem.token_dim if hasattr(cfg.stem, "token_dim") else cfg.encoder.token_dim:
            # In our current design, InputStemConfig already has token_dim,
            # so we want them to match encoder.token_dim.
            if cfg.encoder.token_dim != cfg.stem.token_dim:
                raise ValueError(
                    f"encoder.token_dim ({cfg.encoder.token_dim}) must match "
                    f"stem.token_dim ({cfg.stem.token_dim})."
                )

        if cfg.router.token_dim != cfg.encoder.token_dim:
            raise ValueError(
                f"router.token_dim ({cfg.router.token_dim}) must match "
                f"encoder.token_dim ({cfg.encoder.token_dim})."
            )

        if cfg.router.aux_dim != cfg.encoder.aux_dim:
            raise ValueError(
                f"router.aux_dim ({cfg.router.aux_dim}) must match "
                f"encoder.aux_dim ({cfg.encoder.aux_dim})."
            )

        if cfg.decoder.token_dim != cfg.encoder.token_dim:
            raise ValueError(
                f"decoder.token_dim ({cfg.decoder.token_dim}) must match "
                f"encoder.token_dim ({cfg.encoder.token_dim})."
            )

        if cfg.experts.state_dim != (cfg.router.expert_state_dim or cfg.router.token_dim):
            raise ValueError(
                f"experts.state_dim ({cfg.experts.state_dim}) must match "
                f"router.expert_state_dim ({cfg.router.expert_state_dim or cfg.router.token_dim})."
            )

        if cfg.decoder.expert_state_dim != cfg.experts.state_dim:
            raise ValueError(
                f"decoder.expert_state_dim ({cfg.decoder.expert_state_dim}) must match "
                f"experts.state_dim ({cfg.experts.state_dim})."
            )

        if cfg.decoder.num_experts != cfg.router.num_experts or cfg.decoder.num_experts != cfg.experts.num_experts:
            raise ValueError(
                "decoder.num_experts, router.num_experts, and experts.num_experts must all match."
            )

        # Submodules
        self.stem = InputStem(cfg.stem)
        self.encoder = Encoder(cfg.encoder)
        self.router = GraphRouter(cfg.router)
        self.experts = Experts(cfg.experts)
        self.decoder = Decoder(cfg.decoder)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        x:
            Raw input volume x ∈ R^{B×C_in×D×H×W}.
        key_padding_mask:
            Optional mask over tokens [B, N] used by encoder/router
            to ignore invalid / padded patches.
        return_intermediates:
            If True, include intermediate tensors (tokens, aux, routing,
            expert states, meta) in the output dictionary for debugging
            or analysis.

        Returns
        -------
        out:
            Dictionary with at least:
                - "logits":      model outputs (3D grid or token-wise).
                - "aux_loss":    scalar sum of router regularizers.
                - "lb_loss":     load-balancing regularizer.
                - "entropy_loss":entropy regularizer.

            If return_intermediates=True, also includes:
                - "stem_tokens", "stem_aux", "meta"
                - "enc_tokens", "enc_aux"
                - "routing"       (router dict)
                - "expert_states_router"
                - "expert_states_refined"
        """
        # 1) Input stem
        stem_tokens, stem_aux, meta = self.stem(x)  # [B, N, d_token], [B, N, d_aux]

        # 2) Encoder
        enc_tokens, enc_aux = self.encoder(
            stem_tokens,
            stem_aux,
            key_padding_mask=key_padding_mask,
        )

        # 3) Router
        routing = self.router(
            enc_tokens,
            enc_aux,
            key_padding_mask=key_padding_mask,
        )
        router_expert_states = routing["expert_states"]        # [B, M, D_h]
        routing_probs = routing["probs"]                       # [B, N, M]
        adjacency = routing["adjacency"]                       # [M, M]
        lb_loss = routing["load_balance_loss"]
        entropy_loss = routing["entropy_loss"]
        aux_loss = lb_loss + entropy_loss

        # 4) Experts (graph refinement)
        refined_expert_states = self.experts(router_expert_states, adjacency)

        # 5) Decoder
        logits = self.decoder(
            enc_tokens,
            refined_expert_states,
            routing_probs=routing_probs,
            meta=meta,
        )

        out: Dict[str, Any] = {
            "logits": logits,
            "aux_loss": aux_loss,
            "lb_loss": lb_loss,
            "entropy_loss": entropy_loss,
        }

        if return_intermediates:
            out.update(
                {
                    "stem_tokens": stem_tokens,
                    "stem_aux": stem_aux,
                    "meta": meta,
                    "enc_tokens": enc_tokens,
                    "enc_aux": enc_aux,
                    "routing": routing,
                    "expert_states_router": router_expert_states,
                    "expert_states_refined": refined_expert_states,
                }
            )

        return out
