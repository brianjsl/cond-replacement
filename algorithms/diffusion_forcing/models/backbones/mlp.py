import math
from typing import Optional
import torch
from omegaconf import DictConfig
from einops import rearrange
from .base_backbone import BaseBackbone
from algorithms.common.models.mlp import SimpleMlp


def round_to_pow_2(x):
    log_base_2 = math.log2(x)
    rounded_log = math.ceil(log_base_2)
    power_of_2 = 2**rounded_log

    return power_of_2


class MlpBackbone(BaseBackbone):
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
        if use_causal_mask:
            raise ValueError("Causal mask is not supported for MLP backbone")

        x_dim = x_shape[0]

        num_layers = cfg.num_layers
        self.n_tokens = cfg.n_tokens

        token_dim = x_dim + self.external_cond_emb_dim + self.noise_level_emb_dim
        dim = token_dim * self.n_tokens
        hidden_dim = round_to_pow_2(dim)
        out_dim = x_dim * self.n_tokens

        self.model = SimpleMlp(
            in_dim=dim, out_dim=out_dim, hidden_dim=hidden_dim, n_layers=num_layers
        )

    @property
    def external_cond_emb_dim(self):
        return round_to_pow_2(self.external_cond_dim) if self.external_cond_dim else 0

    @property
    def noise_level_emb_dim(self):
        return 32

    @property
    def time_emb_dim(self):
        return 0

    def forward(
        self,
        x: torch.Tensor,
        noise_levels: torch.Tensor,
        noise_levels_mask: Optional[torch.Tensor] = None,
        external_cond: Optional[torch.Tensor] = None,
    ):
        # x.shape (T, B, C)
        # noise_levels.shape (T, B)
        # noise_levels_mask (T, B)
        # optional external_cond.shape (T, B, C)

        n_tokens, batch_size, _ = x.shape

        if n_tokens != self.n_tokens:
            raise ValueError(
                f"Number of tokens in input ({n_tokens}) does not match the expected value ({self.n_tokens}) specified in architecture"
            )
        x = rearrange(x, "t b c -> b t c")
        noise_levels = rearrange(noise_levels, "t b -> b t")
        if noise_levels_mask is not None:
            noise_levels_mask = rearrange(noise_levels_mask, "t b -> b t")
        if external_cond is not None:
            external_cond = rearrange(external_cond, "t b c -> b t c")

        emb = self.noise_level_pos_embedding(noise_levels, noise_levels_mask)

        if external_cond is not None:
            external_cond = self.external_cond_embedding(external_cond)
            emb = torch.cat((emb, external_cond), dim=-1)
        x = torch.cat((x, emb), dim=-1)
        x = rearrange(x, "b t c -> b (t c)")
        x = self.model(x)
        x = rearrange(x, "b (t c) -> t b c", t=n_tokens)

        return x
