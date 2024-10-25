from functools import partial
from typing import Optional
import torch
from torch import nn
from omegaconf import DictConfig
from einops import rearrange

from algorithms.diffusion_forcing.models.layers.attention import TemporalAttentionBlock
from .base_backbone import BaseBackbone


class Transformer(BaseBackbone):
    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        external_cond_dim: int,
        use_causal_mask=True,
        unknown_noise_level_prob=0.0,
    ):
        super().__init__(
            cfg,
            x_shape,
            external_cond_dim,
            use_causal_mask,
            unknown_noise_level_prob,
        )
        if len(x_shape) > 1:
            raise ValueError("Only 1D series are supported")

        x_dim = x_shape[0]
        size = cfg.network_size
        attn_dim_head = cfg.attn_dim_head
        num_layers = cfg.num_layers
        nhead = cfg.attn_heads

        self.transformer = nn.Sequential(
            *[
                TemporalAttentionBlock(
                    size,
                    nhead,
                    dim_head=attn_dim_head,
                    is_causal=self.use_causal_mask,
                    rotary_emb=self.rotary_time_pos_embedding,
                )
                for _ in range(num_layers)
            ]
        )

        self.init_mlp = nn.Sequential(
            nn.Linear(
                x_dim + self.noise_level_emb_dim + self.external_cond_emb_dim, size
            ),
            nn.SiLU(),
            nn.Linear(size, size),
            nn.SiLU(),
            nn.Linear(size, size),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(size, size),
            nn.SiLU(),
            nn.Linear(size, x_dim),
        )

    @property
    def external_cond_emb_dim(self):
        return self.cfg.network_size if self.external_cond_dim else 0

    @property
    def noise_level_emb_dim(self):
        return self.cfg.network_size

    @property
    def time_emb_dim(self):
        return self.cfg.attn_dim_head

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        noise_levels_mask: Optional[torch.Tensor] = None,
        external_cond: Optional[torch.Tensor] = None,
    ):
        # x.shape (B, T, C)
        # noise_levels.shape (B, T)
        # noise_levels_mask (B, T)
        # optional external_cond.shape (B, T, C)

        emb = self.noise_level_pos_embedding(noise_levels, noise_levels_mask)

        if external_cond is not None:
            external_cond = self.external_cond_embedding(external_cond)
            emb = torch.cat((emb, external_cond), dim=-1)
        x = torch.cat((x, emb), dim=-1)

        x = self.init_mlp(x)
        x = self.transformer(x)
        x = self.out_mlp(x)

        return x

if __name__ == '__main__':
    pass

