from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig, open_dict
from einops import rearrange
from algorithms.diffusion_forcing.models.backbones.base_backbone import BaseBackbone
from algorithms.diffusion_forcing.models.curriculum import Curriculum
from .dit_base import DiTBase


class DiT1D(BaseBackbone):

    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        external_cond_dim: int,
        curriculum: Curriculum,
        use_causal_mask=True,
        unknown_noise_level_prob=0.0,
    ):
        if use_causal_mask:
            raise NotImplementedError(
                "Causal masking is not yet implemented for DiT1D backbone"
            )

        with open_dict(cfg):
            cfg.time_emb_type = None  # no rotary time embedding
        super().__init__(
            cfg,
            x_shape,
            external_cond_dim,
            curriculum,
            use_causal_mask,
            unknown_noise_level_prob,
        )
        if len(x_shape) > 1:
            raise ValueError("Only 1D series are supported")

        hidden_size = cfg.hidden_size
        channels = x_shape[0]

        self.linear_embedder = nn.Linear(channels, hidden_size, bias=True)

        self.dit_base = DiTBase(
            num_patches=None,
            max_temporal_length=max(curriculum.n_tokens),
            out_channels=channels,
            variant="full",
            pos_emb_type=cfg.pos_emb_type,
            hidden_size=hidden_size,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            learn_sigma=False,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize linear_embedder:
        nn.init.xavier_uniform_(self.linear_embedder.weight)
        nn.init.zeros_(self.linear_embedder.bias)

        # Initialize noise level embedding and external condition embedding MLPs:
        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.noise_level_pos_embedding.apply(_mlp_init)
        if self.external_cond_embedding is not None:
            self.external_cond_embedding.apply(_mlp_init)

    @property
    def noise_level_dim(self) -> int:
        return 256

    @property
    def noise_level_emb_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def external_cond_emb_dim(self) -> int:
        return self.cfg.hidden_size if self.external_cond_dim else 0

    @property
    def time_emb_dim(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        noise_levels_mask: Optional[torch.Tensor] = None,
        external_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = rearrange(x, "t b c -> b t c")
        x = self.linear_embedder(x)

        noise_levels = rearrange(noise_levels, "t b -> b t")
        if noise_levels_mask is not None:
            noise_levels_mask = rearrange(noise_levels_mask, "t b -> b t")
        emb = self.noise_level_pos_embedding(noise_levels, noise_levels_mask)

        if external_cond is not None:
            external_cond = rearrange(external_cond, "t b c -> b t c")
            emb = emb + self.external_cond_embedding(external_cond)

        x = self.dit_base(x, emb)
        x = rearrange(x, "b t c -> t b c")
        return x
