from typing import Optional
import torch
from einops import repeat
from algorithms.diffusion_forcing.models.utils import get_einops_wrapped_module
from .attention import AttentionBlock, TemporalAttentionBlock
from .resnet import ResnetBlock

UnetSpatialAttentionBlock = get_einops_wrapped_module(
    AttentionBlock, "b c t h w", "(b t) (h w) c"
)

_UnetTemporalAttentionBlock = get_einops_wrapped_module(
    TemporalAttentionBlock, "b c t h w", "(b h w) t c"
)


class UnetTemporalAttentionBlock(_UnetTemporalAttentionBlock):
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if attn_mask is not None:
            attn_mask = repeat(
                attn_mask, "b t1 t2 -> (b h w) t1 t2", h=x.shape[-2], w=x.shape[-1]
            )
        return super().forward(x, attn_mask)


class UnetSequential(torch.nn.Sequential):
    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ):
        for module in self:
            if isinstance(module, ResnetBlock):
                x = module(x, emb)
            else:
                x = module(x)
        return x
