from functools import partial
from typing import Optional
import torch
from torch import nn
from einops import rearrange
from omegaconf import DictConfig
from algorithms.diffusion_forcing.models.layers.unet3d_layers import (
    UnetSpatialAttentionBlock,
    UnetTemporalAttentionBlock,
    UnetSequential,
)
from algorithms.diffusion_forcing.models.layers.resnet import (
    ResnetBlock,
    Downsample,
    Upsample,
)
from algorithms.diffusion_forcing.models.utils import (
    zero_module,
)
from algorithms.diffusion_forcing.models.curriculum import Curriculum
from .base_backbone import BaseBackbone


class Unet3D(BaseBackbone):
    def __init__(
        self,
        cfg: DictConfig,
        x_shape: torch.Size,
        external_cond_dim: int,
        curriculum: Curriculum,
        use_causal_mask=True,
        unknown_noise_level_prob=0.0,
    ):
        super().__init__(
            cfg,
            x_shape,
            external_cond_dim,
            curriculum,
            use_causal_mask,
            unknown_noise_level_prob,
        )

        dim = cfg.network_size
        init_dim = dim
        channels, resolution, *_ = x_shape
        out_dim = channels
        num_res_blocks = cfg.num_res_blocks
        resnet_block_groups = cfg.resnet_block_groups
        dim_mults = cfg.dim_mults
        attn_resolutions = [resolution // res for res in list(cfg.attn_resolutions)]
        attn_dim_head = cfg.attn_dim_head
        attn_heads = cfg.attn_heads
        use_linear_attn = cfg.use_linear_attn
        use_init_temporal_attn = cfg.use_init_temporal_attn
        init_kernel_size = cfg.init_kernel_size
        dropout = cfg.dropout
        time_emb_type = cfg.time_emb_type

        dims = [*map(lambda m: dim * m, dim_mults)]
        mid_dim = dims[-1]

        emb_dim = self.noise_level_emb_dim + self.external_cond_emb_dim

        init_padding = init_kernel_size // 2

        self.encoder = nn.ModuleList(
            [
                UnetSequential(
                    nn.Conv3d(
                        channels,
                        init_dim,
                        kernel_size=(1, init_kernel_size, init_kernel_size),
                        padding=(0, init_padding, init_padding),
                    ),
                    (
                        UnetTemporalAttentionBlock(
                            dim=init_dim,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                            is_causal=use_causal_mask,
                            rotary_emb=self.rotary_time_pos_embedding,
                        )
                        if use_init_temporal_attn
                        else nn.Identity()
                    ),
                )
            ]
        )
        self.decoder = nn.ModuleList()
        curr_resolution = 1
        curr_channels = init_dim
        encoder_channels = [init_dim]

        block_klass_noise = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            dropout=dropout,
            emb_dim=emb_dim,
            version="v1",
        )
        spatial_attn_klass = partial(
            UnetSpatialAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
        )
        temporal_attn_klass = partial(
            UnetTemporalAttentionBlock,
            heads=attn_heads,
            dim_head=attn_dim_head,
            is_causal=use_causal_mask,
            rotary_emb=self.rotary_time_pos_embedding,
        )

        for idx, ch in enumerate(dims):
            is_last = idx == len(dims) - 1
            use_attn = curr_resolution in attn_resolutions

            for _ in range(num_res_blocks):
                self.encoder.append(
                    UnetSequential(
                        block_klass_noise(curr_channels, ch),
                        (
                            spatial_attn_klass(
                                ch, use_linear=use_linear_attn and not is_last
                            )
                            if use_attn
                            else nn.Identity()
                        ),
                        temporal_attn_klass(ch) if use_attn else nn.Identity(),
                    )
                )
                curr_channels = ch
                encoder_channels.append(ch)

            if not is_last:
                self.encoder.append(
                    UnetSequential(
                        Downsample(ch),
                    )
                )
                curr_resolution *= 2
                encoder_channels.append(ch)

        self.mid_block = UnetSequential(
            block_klass_noise(mid_dim, mid_dim),
            spatial_attn_klass(mid_dim),
            temporal_attn_klass(mid_dim),
            block_klass_noise(mid_dim, mid_dim),
        )

        for idx, ch in enumerate(dims[::-1]):
            is_last = idx == len(dims) - 1
            is_first = idx == 0
            use_attn = curr_resolution in attn_resolutions

            for res_idx in range(num_res_blocks + 1):
                self.decoder.append(
                    UnetSequential(
                        block_klass_noise(curr_channels + encoder_channels.pop(), ch),
                        (
                            spatial_attn_klass(
                                ch, use_linear=use_linear_attn and not is_first
                            )
                            if use_attn
                            else nn.Identity()
                        ),
                        temporal_attn_klass(ch) if use_attn else nn.Identity(),
                        *(
                            [Upsample(ch)]
                            if not is_last and res_idx == num_res_blocks
                            else []
                        )
                    )
                )
                curr_channels = ch
            if not is_last:
                curr_resolution //= 2

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=resnet_block_groups, num_channels=dim, eps=1e-6),
            nn.SiLU(),
            zero_module(
                nn.Conv3d(dim, out_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))
            ),
        )

    @property
    def noise_level_emb_dim(self):
        return self.cfg.network_size * 4

    @property
    def external_cond_emb_dim(self):
        return self.cfg.network_size * 2 if self.external_cond_dim else 0

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
        x = rearrange(x, "t b c h w -> b c t h w")
        noise_levels = rearrange(noise_levels, "t b ...-> b t ...")
        if noise_levels_mask is not None:
            noise_levels_mask = rearrange(noise_levels_mask, "t b ...-> b t ...")
        if external_cond is not None:
            external_cond = rearrange(external_cond, "t b ... -> b t ...")

        emb = self.noise_level_pos_embedding(noise_levels, noise_levels_mask)
        if self.external_cond_embedding is not None:
            if external_cond is None:
                raise ValueError("External condition is required, but not provided.")
            external_cond_emb = self.external_cond_embedding(external_cond)
            emb = torch.cat([emb, external_cond_emb], dim=-1)

        h = x

        hs = []

        for block in self.encoder:
            h = block(h, emb)
            hs.append(h)

        h = self.mid_block(h, emb)

        for block in self.decoder:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, emb)

        x = self.out(h)
        x = rearrange(x, " b c t h w -> t b c h w")

        return x
