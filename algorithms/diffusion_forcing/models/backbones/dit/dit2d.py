
import torch
from torch import nn
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed
from algorithms.diffusion_forcing.models.backbones.base_backbone import BaseBackbone
from algorithms.diffusion_forcing.models.curriculum import Curriculum


class DiT2D(BaseBackbone):
    def __init__(
        self,
        cfg,
        img_shape: torch.Size,
        external_cond_dim: int,
        curriculum: Curriculum,
        use_causal_mask=True,
        unknown_noise_level_prob=0.0,
    ):
        # Ensure causal masking is not enabled for 2D
        if use_causal_mask:
            raise NotImplementedError("Causal masking is not relevant for DiT2D backbone")

        # Adapt configuration for 2D images
        with open_dict(cfg):
            cfg.time_emb_type = None  # No rotary time embedding for images
        super().__init__(
            cfg,
            img_shape,
            external_cond_dim,
            curriculum,
            use_causal_mask=False,  # No causal mask for images
            unknown_noise_level_prob=unknown_noise_level_prob,
        )

        # Patch embedding adapted to 2D images
        self.patch_embed = PatchEmbed(
            img_size=img_shape[1], patch_size=cfg.patch_size, in_chans=img_shape[0], embed_dim=cfg.embed_dim
        )

        # Transformer layers remain the same, but no temporal component
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=cfg.embed_dim, nhead=cfg.num_heads)
            for _ in range(cfg.depth)
        ])

        # Final output projection back to image space
        self.output_projection = nn.Linear(cfg.embed_dim, img_shape[0] * (cfg.patch_size ** 2))

    def forward(self, x, t):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional encoding (if applicable in cfg)
        if hasattr(self, 'positional_encoding'):
            x += self.positional_encoding

        # Process through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Reshape the patches back to image format
        B, N, D = x.shape
        x = rearrange(x, 'b (h w) d -> b d h w', h=int(N ** 0.5))
        x = self.output_projection(x)

        return x
