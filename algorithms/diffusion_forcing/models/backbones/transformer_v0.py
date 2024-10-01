import torch
import torch.nn as nn
import math
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Transformerv0(nn.Module):
    def __init__(
        self,
        cfg,
        x_shape: torch.Size,
        external_cond_dim: int,
        curriculum,
        use_causal_mask=True,
        unknown_noise_level_prob=0.0,
    ):
        x_dim = x_shape[0]
        size = cfg.network_size
        num_layers = cfg.num_layers
        nhead = cfg.attn_heads
        dim_feedforward = cfg.dim_feedforward
        dropout = cfg.dropout

        super(Transformerv0, self).__init__()
        self.use_causal_mask = use_causal_mask
        self.external_cond_dim = external_cond_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        k_embed_dim = size // 2
        self.t_embed = SinusoidalPosEmb(dim=size)
        self.k_embed = SinusoidalPosEmb(dim=k_embed_dim)
        self.init_mlp = nn.Sequential(
            nn.Linear(x_dim + k_embed_dim + external_cond_dim, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )
        self.out = nn.Linear(size, x_dim)

    def forward(self, x, k, t_mask, external_cond=None):
        # x.shape (T, B, C)
        # k.shape (T, B)
        # optional external_cond.shape (T, B, C)

        seq_len, batch_size, _ = x.shape
        k_embed = rearrange(self.k_embed(k.flatten()), "(t b) d -> t b d", t=seq_len)
        x = torch.cat((x, k_embed), dim=-1)
        if external_cond is not None:
            x = torch.cat((x, external_cond), dim=-1)
        x = self.init_mlp(x)
        x = x + self.t_embed(torch.arange(seq_len, device=x.device)[:, None])

        mask = (
            nn.Transformer.generate_square_subsequent_mask(len(x), x.device)
            if self.use_causal_mask
            else None
        )
        x = self.transformer(x, mask=mask, is_causal=self.use_causal_mask)
        x = self.out(x)

        return x


# if __name__ == "__main__":
#     model = Transformerv0(x_dim=10)
#     x = torch.randn(100, 32, 10)
#     k = torch.randint(0, 10, (100, 32))
#     out = model(x, k)
#     print(out.shape)
