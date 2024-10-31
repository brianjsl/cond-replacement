from typing import Optional, Callable, Literal
from collections import namedtuple
from omegaconf import DictConfig
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, reduce

from .backbones.unet3d_v0 import Unet3D as Unet3Dv0
from .backbones.unet3d_v1 import Unet3D as Unet3Dv1
from .backbones.transformer_v1 import Transformer
from .backbones.transformer_v0 import Transformerv0
from .backbones.dit import DiT3D, DiT1D
from .backbones.mlp import MlpBackbone
from .utils import make_beta_schedule, extract
import sys
from tqdm import tqdm

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)


class Diffusion(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        backbone_cfg: DictConfig,
        x_shape: torch.Size,
        external_cond_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.x_shape = x_shape
        self.external_cond_dim = external_cond_dim
        self.timesteps = cfg.timesteps
        self.sampling_timesteps = cfg.sampling_timesteps
        self.beta_schedule = cfg.beta_schedule
        self.schedule_fn_kwargs = cfg.schedule_fn_kwargs
        self.objective = cfg.objective
        self.loss_weighting = cfg.loss_weighting
        self.snr_clip = cfg.snr_clip
        self.cum_snr_decay = cfg.cum_snr_decay
        self.ddim_sampling_eta = cfg.ddim_sampling_eta
        self.clip_noise = cfg.clip_noise

        self.backbone_cfg = backbone_cfg
        self.use_causal_mask = cfg.use_causal_mask
        self.unknown_noise_level_prob = cfg.unknown_noise_level_prob
        self.stabilization_level = cfg.stabilization_level


        self._build_model()
        self._build_buffer()

    def _build_model(self):
        match self.backbone_cfg.name:
            case "unet_v0":
                model_cls = Unet3Dv0
            case "unet_v1":
                model_cls = Unet3Dv1
            case "dit3d":
                model_cls = DiT3D
            case "dit1d":
                model_cls = DiT1D
            case "transformer_v1":
                model_cls = Transformer
            case "mlp":
                model_cls = MlpBackbone
            case "transformer_v0":
                model_cls = Transformerv0
            case _:
                raise ValueError(f"unknown model type {self.model_type}")
        self.model = model_cls(
            cfg=self.backbone_cfg,
            x_shape=self.x_shape,
            external_cond_dim=self.external_cond_dim,
            use_causal_mask=self.use_causal_mask,
            unknown_noise_level_prob=self.unknown_noise_level_prob,
        )

    def _build_buffer(self):
        betas = make_beta_schedule(
            schedule=self.beta_schedule,
            timesteps=self.timesteps,
            zero_terminal_snr=self.objective != "pred_noise",
            **self.schedule_fn_kwargs,
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # sampling related parameters
        assert self.sampling_timesteps <= self.timesteps
        self.is_ddim_sampling = self.sampling_timesteps < self.timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        # if (
        #     self.objective == "pred_noise"
        #     or self.cfg.reconstruction_guidance is not None
        # ):
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # sigma_t in https://arxiv.org/pdf/2010.02502
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # derive loss weight
        # https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)
        clipped_snr = snr.clone()
        clipped_snr.clamp_(max=self.snr_clip)

        register_buffer("clipped_snr", clipped_snr)
        register_buffer("snr", snr)

    def add_shape_channels(self, x):
        return rearrange(x, f"... -> ...{' 1' * len(self.x_shape)}")

    def model_predictions(self, x, k, k_mask=None, external_cond=None):
        model_output = self.model(x, k, k_mask, external_cond)
        model_output = self.model(x, k, k_mask, external_cond)

        if self.objective == "pred_noise":
            pred_noise = torch.clamp(model_output, -self.clip_noise, self.clip_noise)
            x_start = self.predict_start_from_noise(x, k, pred_noise)

        elif self.objective == "pred_x0":
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, k, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, k, v)
            pred_noise = self.predict_noise_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred

    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def predict_noise_from_start(self, x_k, k, x0):
        # return (
        #     extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        # ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (x_k - extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x0) / extract(
            self.sqrt_one_minus_alphas_cumprod, k, x_k.shape
        )

    def predict_v(self, x_start, k, noise):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            extract(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )

    def q_mean_variance(self, x_start, k):
        mean = extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, k, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, k, x_start.shape)
        return mean, variance, log_variance

    def q_posterior(self, x_start, x_k, k):
        posterior_mean = (
            extract(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + extract(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = extract(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, k, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        return (
            extract(self.sqrt_alphas_cumprod, k, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, k, x_start.shape) * noise
        )

    def p_mean_variance(self, x, k, k_mask, external_cond=None):
        model_pred = self.model_predictions(
            x=x, k=k, k_mask=k_mask, external_cond=external_cond
        )
        x_start = model_pred.pred_x_start
        return self.q_posterior(x_start=x_start, x_k=x, k=k)

    def compute_loss_weights(
        self,
        k: torch.Tensor,
        loss_weighting: Literal["min_snr", "fused_min_snr", "uniform"],
    ) -> torch.Tensor:
        if loss_weighting == "uniform":
            return torch.ones_like(k)

        snr = self.snr[k]
        clipped_snr = self.clipped_snr[k]
        normalized_clipped_snr = clipped_snr / self.snr_clip
        normalized_snr = snr / self.snr_clip

        if loss_weighting == "min_snr":
            # min SNR reweighting
            match self.objective:
                case "pred_noise":
                    return clipped_snr / snr
                case "pred_x0":
                    return clipped_snr
                case "pred_v":
                    return clipped_snr / (snr + 1)

        elif loss_weighting in {"min_snr", "fused_min_snr"}:
            if loss_weighting == "fused_min_snr":

                def compute_cum_snr(reverse: bool = False):
                    new_normalized_clipped_snr = (
                        normalized_clipped_snr.flip(1)
                        if reverse
                        else normalized_clipped_snr
                    )
                    cum_snr = torch.zeros_like(new_normalized_clipped_snr)
                    for t in range(0, k.shape[1]):
                        if t == 0:
                            cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                        else:
                            cum_snr[:, t] = (
                                self.cum_snr_decay * cum_snr[:, t - 1]
                                + (1 - self.cum_snr_decay)
                                * new_normalized_clipped_snr[:, t]
                            )
                    cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                    return cum_snr.flip(1) if reverse else cum_snr

                if self.use_causal_mask:
                    cum_snr = compute_cum_snr()
                else:
                    # bi-directional cum_snr when not using causal mask
                    cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
                    cum_snr *= 0.5
                clipped_fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (
                    1 - normalized_clipped_snr
                )
                fused_snr = 1 - (1 - cum_snr * self.cum_snr_decay) * (
                    1 - normalized_snr
                )
                clipped_snr = clipped_fused_snr * self.snr_clip
                snr = fused_snr * self.snr_clip
            match self.objective:
                case "pred_noise":
                    return clipped_snr / snr
                case "pred_x0":
                    return clipped_snr
                case "pred_v":
                    return clipped_snr / (snr + 1)
                case _:
                    raise ValueError(f"unknown objective {self.objective}")
        else:
            raise ValueError(f"unknown loss weighting strategy {loss_weighting}")

    def forward(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        k: torch.Tensor,
    ):
        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        noised_x = self.q_sample(x_start=x, k=k, noise=noise)
        model_pred = self.model_predictions(
            x=noised_x, k=k, external_cond=external_cond
        )

        pred = model_pred.model_out
        x_pred = model_pred.pred_x_start

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x
        elif self.objective == "pred_v":
            target = self.predict_v(x, k, noise)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(pred, target.detach(), reduction="none")

        loss_weight = self.compute_loss_weights(k, self.loss_weighting)
        loss_weight = self.add_shape_channels(loss_weight)
        loss = loss * loss_weight

        return x_pred, loss

    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    def ddim_idx_to_pyramid_noise_level(
        self,
        indices: torch.Tensor,
        start_frame: int,
        end_frame: int,
        start_exponent: float,
        end_exponent: float,
    ):
        indices = indices.float()
        exponents = torch.ones_like(indices[0])
        linear_exponents = torch.linspace(
            start_exponent,
            end_exponent,
            steps=end_frame - start_frame + 1,
            device=indices.device,
        )
        exponents[start_frame : end_frame + 1] = linear_exponents
        real_steps = (
            self.timesteps
            * (indices / self.sampling_timesteps).pow(exponents.unsqueeze(0))
            - 1
        ).long()
        return real_steps

    def sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        ukn_noise_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        rg_fns: Optional[list] = None,
    ):
        # real_steps = torch.linspace(
        #     -1, self.timesteps - 1, steps=self.sampling_timesteps + 1, device=x.device
        # ).long()

        # # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
        # curr_noise_level = real_steps[curr_noise_level]
        # next_noise_level = real_steps[next_noise_level]

        if ukn_noise_mask is not None and ukn_noise_mask.any():
            raise ValueError("New compositionality shall not involve unknown noise")

        if self.is_ddim_sampling:
            return self.ddim_sample_step(
                x=x,
                external_cond=external_cond,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                ukn_noise_mask=ukn_noise_mask,
                guidance_fn=guidance_fn,
                rg_fns = rg_fns
            )

        # FIXME: temporary code for checking ddpm sampling
        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            x=x,
            external_cond=external_cond,
            curr_noise_level=curr_noise_level,
            ukn_noise_mask=ukn_noise_mask,
            guidance_fn=guidance_fn,
            rg_fns = rg_fns
        )
        
    def ddpm_sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        ukn_noise_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_scale: Optional[float] = 1.0,
        rg_fns: Optional[list] = None,
    ):

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            k=clipped_curr_noise_level,
            k_mask=ukn_noise_mask,
            external_cond=external_cond,
        )

        if guidance_fn is not None:
            with torch.enable_grad():
                guidance_loss = guidance_fn(
                    xk=x
                )
                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
            model_mean += guidance_scale * ()
            


        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise

        # only update frames where the noise level decreases
        return torch.where(self.add_shape_channels(curr_noise_level == -1), x, x_pred)

    def ddim_sample_step(
        self,
        x: torch.Tensor,
        external_cond: Optional[torch.Tensor],
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        ukn_noise_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        rg_fns: Optional[list] = None,
    ):

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = self.alphas_cumprod[clipped_curr_noise_level]
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),
            self.alphas_cumprod[next_noise_level],
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            self.ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha)
        alpha_next = self.add_shape_channels(alpha_next)
        c = self.add_shape_channels(c)
        sigma = self.add_shape_channels(sigma)

        if guidance_fn is not None or rg_fns is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    x=x,
                    k=clipped_curr_noise_level,
                    k_mask=ukn_noise_mask,
                    external_cond=external_cond,
                )

                if guidance_fn:
                    guidance_loss = guidance_fn(
                        xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                    )

                    grad = -torch.autograd.grad(
                        guidance_loss,
                        x,
                    )[0].detach()

                for mixture in tqdm(rg_fns):
                    mixture_loss = mixture(
                        xk = x, pred_x0 = model_pred.pred_x_start, alpha_cumprod = alpha
                    )
                    if guidance_fn:
                        grad += -torch.autograd.grad(
                            mixture_loss,
                            x,
                        )[0].detach()
                    else:
                        grad = -torch.autograd.grad(
                            mixture_loss,
                            x,
                        )[0].detach()

                    grad = torch.nan_to_num(grad, nan=0.0)

                    torch.cuda.empty_cache()

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,  # to avoid NaN from zero terminal SNR
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )

        else:
            model_pred = self.model_predictions(
                x=x,
                k=clipped_curr_noise_level,
                k_mask=ukn_noise_mask,
                external_cond=external_cond,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise

        # only update frames where the noise level decreases
        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask),
            x,
            x_pred,
        )

        return x_pred

    def estimate_noise_level(self, x, mu=None):
        # x ~ ( B, T, C, ...)
        if mu is None:
            mu = torch.zeros_like(x)
        x = x - mu
        mse = reduce(x**2, "b t ... -> b t", "mean")
        ll_except_c = -self.log_one_minus_alphas_cumprod[None, None] - mse[
            ..., None
        ] * self.alphas_cumprod[None, None] / (1 - self.alphas_cumprod[None, None])
        k = torch.argmax(ll_except_c, -1)
        return k
