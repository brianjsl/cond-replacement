from typing import Literal
import math
import torch
from torch import nn
from einops import rearrange, parse_shape
import matplotlib.pyplot as plt


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    shape = t.shape
    out = a[t]
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def make_beta_schedule(
    schedule: Literal["cosine", "sigmoid", "sd", "linear", "alphas_cumprod_linear"],
    shift: float = 1.0,
    clip_min: float = 1e-9,
    zero_terminal_snr: bool = True,
    **kwargs,
):
    schedule_fn = {
        "alphas_cumprod_linear": alphas_cumprod_linear_schedule,
        "cosine": cosine_schedule,
        "sigmoid": sigmoid_schedule,
        "sd": sd_schedule,
        "linear": beta_linear_schedule,
    }[schedule]
    alphas_cumprod = schedule_fn(**kwargs)
    if schedule != "cosine" and zero_terminal_snr:
        # cosine schedule already enforces zero terminal SNR
        alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)
    if shift != 1.0:
        alphas_cumprod = shift_beta_schedule(alphas_cumprod, shift)
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    alphas = torch.cat([alphas_cumprod[0:1], alphas])
    betas = 1 - alphas
    return torch.clip(betas, clip_min, 1.0)


def cosine_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def alphas_cumprod_linear_schedule(timesteps: int) -> torch.Tensor:
    """
    linear schedule
    as proposed in https://arxiv.org/abs/2301.10972
    """
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    return (1 - t)[1:]


def beta_linear_schedule(
    timesteps: int, start: float = 0.0001, end: float = 0.02
) -> torch.Tensor:
    """
    linear schedule
    as proposed in https://arxiv.org/abs/2006.11239 (original DDPM paper)
    """
    betas = torch.linspace(start, end, timesteps, dtype=torch.float64)
    return (1 - betas).cumprod(dim=0)


def sigmoid_schedule(timesteps, start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    return alphas_cumprod[1:]


def sd_schedule(timesteps, start=0.00085, end=0.0120):
    """
    stable diffusion's noise schedule
    https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/util.py#L21
    """
    betas = torch.linspace(start**0.5, end**0.5, timesteps, dtype=torch.float64) ** 2
    alphas_cumprod = (1 - betas).cumprod(dim=0)
    return alphas_cumprod


def shift_beta_schedule(alphas_cumprod: torch.Tensor, shift: float):
    """
    scale alphas_cumprod so that SNR is multiplied by shift ** 2
    """
    snr_scale = shift**2

    return (snr_scale * alphas_cumprod) / (
        snr_scale * alphas_cumprod + 1 - alphas_cumprod
    )


def enforce_zero_terminal_snr(alphas_cumprod):
    """
    enforce zero terminal SNR following https://arxiv.org/abs/2305.08891
    returns betas
    """
    alphas_cumprod_sqrt = torch.sqrt(alphas_cumprod)

    # store old values
    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
    # shift so last timestep is zero
    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
    # scale so first timestep is back to original value
    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / alphas_cumprod_sqrt[0]
    # convert to betas
    alphas_cumprod = alphas_cumprod_sqrt**2
    assert alphas_cumprod[-1] == 0, "terminal SNR not zero"
    return alphas_cumprod


class EinopsWrapper(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, module: nn.Module):
        super().__init__()
        self.module = module
        self.from_shape = from_shape
        self.to_shape = to_shape

    def forward(self, x: torch.Tensor, *args, **kwargs):
        axes_lengths = parse_shape(x, pattern=self.from_shape)
        x = rearrange(x, f"{self.from_shape} -> {self.to_shape}")
        x = self.module(x, *args, **kwargs)
        x = rearrange(x, f"{self.to_shape} -> {self.from_shape}", **axes_lengths)
        return x


def get_einops_wrapped_module(module, from_shape: str, to_shape: str):
    class WrappedModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.wrapper = EinopsWrapper(from_shape, to_shape, module(*args, **kwargs))

        def forward(self, x: torch.Tensor, *args, **kwargs):
            return self.wrapper(x, *args, **kwargs)

    return WrappedModule


def plot_beta_schedules(
    schedules: list[
        Literal["cosine", "sigmoid", "sd", "linear", "alphas_cumprod_linear"]
    ],
    timesteps: int,
    schedule_kwargs: list[dict],
    path: str,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for schedule, kwargs in zip(schedules, schedule_kwargs):
        betas = torch.cat(
            [
                torch.tensor([0.0]),
                make_beta_schedule(schedule, timesteps=timesteps, **kwargs),
            ]
        )
        alphas_cumprod = (1 - betas).cumprod(dim=0)
        log_snrs = torch.log(alphas_cumprod / (1 - alphas_cumprod))
        times = torch.arange(len(alphas_cumprod))

        # Plot logSNR to the first subplot
        ax1.plot(times, log_snrs, label=f"{schedule} {kwargs}")

        # Plot alphas_cumprod to the second subplot
        ax2.plot(times, alphas_cumprod, label=f"{schedule} {kwargs}")

    # Setup for logSNR plot
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\log{SNR}$")
    ax1.legend()
    ax1.set_title(r"$\log{SNR}$ over Time")

    # Setup for alphas_cumprod plot
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\alpha$")
    ax2.legend()
    ax2.set_title(r"$\alpha$ over Time")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    plot_beta_schedules(
        [
            "cosine",
            "sigmoid",
            "sd",
            "sd",
            "alphas_cumprod_linear",
            "alphas_cumprod_linear",
            "linear",
        ],
        timesteps=1000,
        schedule_kwargs=[
            {"s": 0.008},
            {"start": -3, "end": 3, "tau": 1},
            {"start": 0.00085, "end": 0.0120},
            {"start": 0.0015, "end": 0.0195},
            {"shift": 0.3},
            {"shift": 0.5},
            {"start": 0.0001, "end": 0.02},
        ],
        path="beta_schedules.png",
    )
