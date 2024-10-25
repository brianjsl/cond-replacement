"""
Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py
"""

from typing import Optional, Tuple
import math
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from diffusers.models.embeddings import TimestepEmbedding
from rotary_embedding_torch import RotaryEmbedding
from rotary_embedding_torch.rotary_embedding_torch import rotate_half
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from algorithms.diffusion_forcing.models.curriculum import Curriculum


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class StochasticUnknownTimesteps(Timesteps):
    def __init__(
        self,
        num_channels: int,
        p: float = 1.0,
    ):
        super().__init__(num_channels)
        self.unknown_token = (
            nn.Parameter(torch.randn(1, num_channels)) if p > 0.0 else None
        )
        self.p = p

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        t_emb = super().forward(timesteps)
        # if p == 0.0 - return original embeddings both during training and inference
        if self.p == 0.0:
            return t_emb

        # training or mask is None - randomly replace embeddings with unknown token with probability p
        # (mask can only be None for logging training visualization when using latents)
        # or if p == 1.0 - always replace embeddings with unknown token even during inference)
        if self.training or self.p == 1.0 or mask is None:
            mask = torch.rand(t_emb.shape[:-1], device=t_emb.device) < self.p
            mask = mask[..., None].expand_as(t_emb)
            return torch.where(mask, self.unknown_token, t_emb)

        # # inference with p < 1.0 - replace embeddings with unknown token only for masked timesteps
        # if mask is None:
        #     assert False, "mask should be provided when 0.0 < p < 1.0"
        mask = mask[..., None].expand_as(t_emb)
        return torch.where(mask, self.unknown_token, t_emb)


class StochasticTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_embed_dim: int,
        p: float = 1.0,
    ):
        super().__init__()
        self.timesteps = StochasticUnknownTimesteps(dim, p)
        self.embedding = TimestepEmbedding(dim, time_embed_dim)

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.embedding(self.timesteps(timesteps, mask))


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D or 2-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] or [N x M x dim] Tensor of positional embeddings.
    """
    if len(timesteps.shape) not in [1, 2]:
        raise ValueError("Timesteps should be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[..., None].float() * emb

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[..., half_dim:], emb[..., :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class FlexibleRotaryEmbedding(nn.Module):
    """
    RoPE that can be rescaled during training and inference,
    to allow flexible length sequences.
    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()
        self.dim = dim
        self.model = None
        self.current_seq_len = None
        self._update_model()

    def _update_model(self):

        device = self.model.device if self.model is not None else None
        
        self.model = RotaryEmbedding(
            self.dim,
        )

        if device is not None:
            self.model.to(device)

    def rotate_queries_or_keys(self, queries_or_keys: torch.Tensor) -> torch.Tensor:
        self._update_model()
        return self.model.rotate_queries_or_keys(queries_or_keys)


class RotaryEmbeddingND(nn.Module):
    """
    Minimal Axial RoPE generalized to N dimensions.
    """

    def __init__(
        self,
        dims: Tuple[int, ...],
        sizes: Tuple[int, ...],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        """
        Args:
            dims: the number of dimensions for each axis.
            sizes: the maximum length for each axis.
        """
        super().__init__()
        self.n_dims = len(dims)
        self.dims = dims
        self.theta = theta
        self.flatten = flatten

        Colon = slice(None)
        all_freqs = []
        for i, (dim, seq_len) in enumerate(zip(dims, sizes)):
            freqs = self.get_freqs(dim, seq_len)
            all_axis = [None] * len(dims)
            all_axis[i] = Colon
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice].expand(*sizes, dim))
        all_freqs = torch.cat(all_freqs, dim=-1)
        if flatten:  # flatten all but the last dimension
            all_freqs = rearrange(all_freqs, "... d -> (...) d")
        self.register_buffer("freqs", all_freqs, persistent=False)

    def get_freqs(self, dim: int, seq_len: int) -> torch.Tensor:
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        pos = torch.arange(seq_len, dtype=freqs.dtype)
        freqs = einsum("..., f -> ... f", pos, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: a [... x N x ... x D] if flatten=False, [... x (N x ...) x D] if flatten=True tensor of queries or keys.
        Returns:
            a tensor of rotated queries or keys. (same shape as x)
        """
        # slice the freqs to match the input shape
        seq_shape = x.shape[-2:-1] if self.flatten else x.shape[-self.n_dims - 1 : -1]
        slice_tuple = tuple(slice(0, seq_len) for seq_len in seq_shape)
        freqs = self.freqs[slice_tuple]
        return x * freqs.cos() + rotate_half(x) * freqs.sin()


class RotaryEmbedding3D(RotaryEmbeddingND):
    """
    RoPE3D for Video Transformer.
    Handles tensors of shape [B x T x H x W x C] or [B x (T x H x W) x C].
    """

    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding3D requires even dim"
        dim //= 2

        # if dim is not divisible by 3,
        # split into 3 dimensions such that height and width have the same number of frequencies
        match dim % 3:
            case 0:
                dims = (dim // 3,) * 3
            case 1:
                dims = (dim // 3 + 1, dim // 3, dim // 3)
            case 2:
                dims = (dim // 3, dim // 3 + 1, dim // 3 + 1)

        super().__init__(tuple(d * 2 for d in dims), sizes, theta, flatten)
