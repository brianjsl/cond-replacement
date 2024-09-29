from abc import abstractmethod, ABC
from functools import partial
from typing import Optional, List
import torch
from torch import nn
from einops import rearrange
from diffusers.models.embeddings import TimestepEmbedding as ExternalCondEmbedding
from omegaconf import DictConfig
from algorithms.replacement_diffusion.models.layers.embeddings import (
    StochasticTimeEmbedding,
    FlexibleRotaryEmbedding,
)


class BaseBackbone(ABC, nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        external_cond_dim: int,
        use_causal_mask=True,
        unknown_noise_level_prob=0.0,
    ):

        super().__init__()

        self.cfg = cfg
        self.external_cond_dim = external_cond_dim
        self.use_causal_mask = use_causal_mask
        self.unknown_noise_level_prob = unknown_noise_level_prob
        self.x_shape = x_shape

        time_emb_type = cfg.time_emb_type

        self.noise_level_pos_embedding = StochasticTimeEmbedding(
            dim=self.noise_level_dim,
            time_embed_dim=self.noise_level_emb_dim,
            p=unknown_noise_level_prob,
        )
        self.external_cond_embedding = (
            ExternalCondEmbedding(external_cond_dim, self.external_cond_emb_dim)
            if self.external_cond_dim
            else None
        )

        self.rotary_time_pos_embedding = (
            FlexibleRotaryEmbedding(dim=self.time_emb_dim)
            if time_emb_type == "rotary"
            else None
        )

    @property
    def noise_level_dim(self):
        return max(self.noise_level_emb_dim // 4, 32)

    @property
    @abstractmethod
    def noise_level_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def external_cond_emb_dim(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def time_emb_dim(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        noise_levels_mask: Optional[torch.Tensor] = None,
        external_cond: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError
