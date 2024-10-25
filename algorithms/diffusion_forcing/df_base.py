"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional, Callable, Tuple, Any, Sequence
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat, einsum
import copy
import sys


from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.diffusion_forcing.models import Diffusion
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion


class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.x_shape = cfg.x_shape #shape of input, eg. [1] for numerical, [3] for image, [3, 32, 32] for video
        self.frame_stack = cfg.frame_stack
        self.frame_skip = cfg.frame_skip

        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack

        self.guidance_scale = cfg.guidance_scale

        self.external_cond_dim = (
            cfg.external_cond_dim
            * cfg.frame_stack
            * (cfg.frame_skip if cfg.external_cond_stack else 1)
        )
        self.logging = cfg.logging

        self.unknown_noise_level_prob = cfg.diffusion.unknown_noise_level_prob #probability of unknown noise level for stochastic timestep embedding

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        self.replacement = cfg.replacement

        cfg.diffusion.cum_snr_decay = cfg.diffusion.cum_snr_decay ** (
            self.frame_stack * cfg.frame_skip
        )

        self.validation_multiplier = cfg.validation_multiplier
        self.validation_step_outputs = []

        match cfg.loss:
            case "mse":
                self.loss_fn = nn.functional.mse_loss
            case "l1":
                self.loss_fn = nn.functional.l1_loss

        super().__init__(cfg)

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    def _build_model(self):
        self.diffusion_model = Diffusion(
            cfg=self.cfg.diffusion,
            backbone_cfg=self.cfg.backbone,
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
        )
        self.register_data_mean_std(self.cfg.data_mean, self.cfg.data_std)

    def configure_optimizers(self):
        transition_params = list(self.diffusion_model.parameters())
        optimizer_dynamics = torch.optim.AdamW(
            transition_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )

        return optimizer_dynamics

    # ---------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------

    @property
    def is_full_sequence(self) -> bool:
        """
        Whether the model is a standard full sequence diffusion model.
        """
        return (
            self.cfg.noise_level == "random_uniform" and not self.cfg.noiseless_context
        )

    # ---------------------------------------------------------------------
    # Length-related Properties
    # NOTE: "Frame" and "Token" should be distinguished carefully.
    # "Frame" refers to original unit of data loaded from dataset.
    # "Token" refers to the unit of data processed by the diffusion model.
    # There is no guarantee that n_frames == n_tokens * frame_stack.
    # ---------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """
        Get the number of frames to be processed.
        """
        return self.cfg.n_frames

    @property
    def n_tokens(self) -> int:
        '''
        Get the number of tokens inthe model. 
        '''
        return self.n_frames // self.frame_stack

    @property
    def n_context_tokens(self) -> int:
        """
        Get the number of context tokens for the model.
        """
        return self.cfg.context_frames / self.frame_stack

    @property
    def n_context_frames(self) -> int:
        """
        Get the number of context frames for the model.
        """
        return self.cfg.context_frames

    # ---------------------------------------------------------------------
    # Data Preprocessing
    # ---------------------------------------------------------------------

    def on_after_batch_transfer(self, batch: tuple, dataloader_idx: int) -> tuple:
        """
        preprocess the batch for training and validation.

        Args:
            batch (tuple): a tuple of tensors
            dataloader_idx (int): dataloader index
        Returns:
            xs: tensor of shape (batch_size, n_tokens, *stacked_x_shape)
            conditions: optional tensor of shape (batch_size, n_tokens, d)
            masks: tensor of shape (batch_size, n_tokens, frame_stack)
        """

        xs = self._normalize_x(batch["xs"])
        conditions = None

        if self.external_cond_dim != 0:

            conditions = rearrange(
                batch["conds"],
                "b (n t fs) d -> (b n) t (fs d)",
                t=self.n_tokens,
                fs=self.frame_stack,
            )

            if conditions.shape[-1] != self.external_cond_dim:
                raise ValueError(
                    f"Expected external condition dim {self.external_cond_dim}, got {conditions.shape[-1]}."
                )

        xs = rearrange(
            xs,
            "b (n t fs) c ... -> (b n) t (fs c) ...",
            fs=self.frame_stack,
            t=self.n_tokens,
        )

        if "masks" in batch:
            masks = rearrange(
                batch["masks"],
                "b (n t fs) ... -> (b n) t fs ...",
                fs=self.frame_stack,
                t=self.n_tokens,
            )
        else:
            masks = torch.ones(*xs.shape[:2], self.frame_stack).bool().to(self.device)

        return xs, conditions, masks

    def _process_conditions(self, conditions):
        return conditions

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    def training_step(self, batch, batch_idx, namespace="training") -> STEP_OUTPUT:
        xs, conditions, masks, *_ = batch
        conditions = self._process_conditions(conditions)
        batch_size, n_tokens, *_ = xs.shape
        xs_pred, loss = self.diffusion_model(
            xs,
            conditions,
            k=self._get_training_noise_levels(xs, masks),
        )

        if self.cfg.noiseless_context:
            masks[:, : self.n_context_frames] = False

        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )

        xs = self._unstack_and_unnormalize(xs)
        xs_pred = self._unstack_and_unnormalize(xs_pred)

        output_dict = {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

        return output_dict

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ) -> None:
        # FIXME: which should come first?
        # manually warm up lr without a scheduler
        if self.trainer.global_step < self.cfg.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.cfg.warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg.lr

        # update params
        optimizer.step(closure=optimizer_closure)

    # ---------------------------------------------------------------------
    # Validation & Test
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks, *_ = batch

        xs_pred, _ = self.predict_sequence(
            xs[:, : self.n_context_tokens],
            xs.shape[1],
            conditions,
            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
        )

        # FIXME: loss
        loss = self.loss_fn(xs_pred, xs, reduction="none")
        loss = self._reweight_loss(loss, masks).detach().cpu()

        xs = self._unstack_and_unnormalize(xs).detach().cpu()
        xs_pred = self._unstack_and_unnormalize(xs_pred).detach().cpu()

        self.validation_step_outputs.append((xs, xs_pred, loss))

        return loss

    def on_validation_epoch_end(self, namespace="validation") -> None:
        # log validation metrics here
        pass

        self.validation_step_outputs = []

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs, namespace="test")

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------

    def predict_sequence(
        self,
        context: torch.Tensor,
        length: Optional[int] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        rg_monte_carlo: bool = False,
        monte_carlo_n: int = 20,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence using the model, using context tokens at the front.
        Args:
            context: shape (batch_size, context_len, *self.x_stacked_shape)
            length: number of tokens to sample.
            conditions: external conditions for sampling, e.g. action or text, optional
            guidance_fn: guidance function for sampling, optional
            reconstruction_guidance: the scale of reconstruction guidance (from Video Diffusion Models Ho. et al.)
            rg_monte_carlo: whether to use monte carlo guidance
            monte_carlo_n: number of monte carlo samples to use for reconstruction guidance
        Returns:
            xs_pred: shape (batch_size, length, *self.x_stacked_shape)
        """
        if length is None:
            length = self.n_tokens 

        xs_pred = context
        batch_size, context_len, _ = context.shape

        if conditions is not None:
            if conditions.shape[1] < length:
                raise ValueError(
                    f"conditions length is expected to be at least the length {length} but got {conditions.shape[1]}."
                )

        context_mask = torch.ones_like(context, dtype=torch.bool).to(context.device)
        pad = torch.zeros([batch_size, length - context_len, *self.x_stacked_shape], dtype=torch.bool).to(context.device)
        context_mask = torch.concat([context_mask, pad], 1)
        # base context noise level, -1 for GT, stabilization_level - 1 for generated tokens

        new_pred = self._sample_sequence(
            batch_size,
            length=length,
            context=context,
            context_mask=context_mask,
            conditions=conditions,
            guidance_fn=guidance_fn,
            reconstruction_guidance=reconstruction_guidance,
            rg_monte_carlo=rg_monte_carlo,
            monte_carlo_n=monte_carlo_n
        )
        xs_pred = torch.cat([xs_pred, new_pred[:, -context_len:]], 1)
        return xs_pred

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        context_guidance: Optional[torch.Tensor] = None,
        rg_monte_carlo: bool = False,
        monte_carlo_n: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of length can be provided along with a mask to achieve conditioning.

        Args:
            batch_size: batch size
            length: total sequence length
            context: context token to condition on. shape (length, batch_size, *self.x_stacked_shape)
            context_mask: Entries are True for context, False otherwise. Same shape as context
            conditions: conditions external conditions for sampling
            guidance_fn: guidance function for sampling
            reconstruction_guidance: the scale of reconstruction guidance for True entries in context_mask
            context_guidance: context guidance from diffusion forcing 2, must be a sequence of floats that sum to 1. nth entry correspond to the weight to the nth context_k
            rg_monte_carlo: if True, use monte carlo sampling for reconstruction guidance
            monte_carlo_n: number of monte carlo samples for reconstruction guidance
        Returns:
            xs_pred: shape (length, batch_size, *self.x_stacked_shape)
            hist: if True, history of xs_pred (m, length, batch_size, *self.x_stacked_shape) on cpu. None otherwise.
        """
        if length is None:
            length = self.n_tokens

        if rg_monte_carlo:
            if reconstruction_guidance == 0:
                raise ValueError(
                    f"RG must be positive but got {reconstruction_guidance}."
                ) 
            if monte_carlo_n == 0:
                raise ValueError(
                    f"Monte Carlo n must be positive but got {monte_carlo_n}."
                )
            if self.cfg.conditional:
                raise ValueError(
                    "Monte Carlo RG not supported for conditional model."
                )

        x_shape = self.x_stacked_shape

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[0] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[0]}."
                )
            if tuple(context.shape[2:]) != tuple(x_shape):
                raise ValueError(
                    f"context shape not compatible with x_stacked_shape {x_shape}."
                )

        if context_mask is not None:
            if context is None:
                raise ValueError("context must be provided if context_mask is given. ")
            if context.shape != context_mask.shape:
                raise ValueError("context and context_mask must have the same shape.")

        # create intial xs_pred with noise
        xs_pred = torch.randn((batch_size, length, *x_shape), device=self.device)
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)

        # replace mask starts with context mask and will be updated with newly diffused tokens
        if context_mask is not None:
            replace_mask = context_mask.clone()

        if context is None:
            # create empty context and zero replacement mask
            context = torch.zeros_like(xs_pred)
            context_mask = torch.zeros_like(context, dtype=torch.bool)
            replace_mask = torch.zeros_like(xs_pred, dtype=torch.bool)

        # replace xs_pred's context frames with context
        xs_pred = torch.where(replace_mask, context, xs_pred)

        # generate scheduling matrix
        # fixme: is mask's reduction over b correct or shall we add some value check?
        scheduling_matrix = self._generate_scheduling_matrix(
            length, 
            False
        )

        scheduling_matrix = scheduling_matrix.to(self.device)
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=batch_size)

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]

            # stabilization, treat certain tokens as noisy
            scaled_xs_pred = self.diffusion_model.q_sample(
                xs_pred,
                from_noise_levels,  # notice entries with -1 will not be handled correctly
                noise=(
                    None
                    if self.replacement == "noisy_scale"
                    else torch.zeros_like(xs_pred)
                ),
            )

            # creates a backup with all clean tokens unscaled
            xs_pred_unstablized = xs_pred.clone()
            xs_pred = torch.where(extended_st_mask, scaled_xs_pred, xs_pred)

            # record value if return history
            if return_hist:
                hist.append(xs_pred.clone())

            # add reconstruction_guidance
            if reconstruction_guidance > 0 and context is not None:
                def composed_guidance_fn(
                    xk: torch.Tensor, pred_x0: torch.Tensor, alpha_cumprod: torch.Tensor
                ) -> torch.Tensor:

                    if rg_monte_carlo:
                        likelihood = 0
                        self.diffusion_model.ddim_sampling_eta = self.cfg.diffusion.rg_eta

                        for _ in range(monte_carlo_n):
                            sample_pred = xk
                            for alpha in range(m, scheduling_matrix.shape[0]-1):
                                from_noise_levels = scheduling_matrix[alpha]
                                to_noise_levels = scheduling_matrix[alpha + 1]
                                sample_pred = self.diffusion_model.sample_step(
                                    sample_pred,
                                    conditions,
                                    from_noise_levels,
                                    to_noise_levels,
                                    guidance_fn=guidance_fn,
                                )

                            loss = (
                                self.loss_fn(sample_pred, context, reduction="none")
                                * alpha_cumprod.sqrt()
                            )

                            loss = torch.sum(
                                loss
                                * replace_mask
                                / replace_mask.sum(dim=0, keepdim=True).clamp(min=1),
                            )
                            likelihood += -reconstruction_guidance * 0.5 * loss
                            likelihood /= monte_carlo_n

                        self.diffusion_model.ddim_sampling_eta = self.cfg.diffusion.ddim_sampling_eta
                    else:
                        loss = (
                            self.loss_fn(pred_x0, context, reduction="none")
                            * alpha_cumprod.sqrt()
                        )
                        # scale inversely proportional to the number of context frames
                        loss = torch.sum(
                            loss
                            * replace_mask
                            / replace_mask.sum(dim=0, keepdim=True).clamp(min=1),
                        )
                        likelihood = -reconstruction_guidance * 0.5 * loss

                    if guidance_fn is not None:
                        likelihood += guidance_fn(
                            xk=xk, pred_x0=pred_x0, alpha_cumprod=alpha_cumprod
                        )
                    return likelihood
            else:
                composed_guidance_fn = guidance_fn

            # update xs_pred by DDIM or DDPM sampling
            # input frames within the sliding window
            xs_pred = self.diffusion_model.sample_step(
                xs_pred,
                conditions,
                from_noise_levels,
                to_noise_levels,
                guidance_fn=composed_guidance_fn,
            )

            if context_guidance is not None:
                _xs_pred = xs_pred.clone()
                xs_pred = rearrange(xs_pred, "(b cg) t ... -> b cg t ...", cg=len_cg)
                xs_pred = einsum(xs_pred, context_guidance, "b cg t ..., cg -> b t ...")
                xs_pred = repeat(xs_pred, "b t ... -> (b cg) t ...", cg=len_cg)
                xs_pred = torch.where(context_mask, _xs_pred, xs_pred)

            if self.replacement is not None:
                # revert to noise level -1 for context frames
                xs_pred = torch.where(replace_mask, xs_pred_unstablized, xs_pred)

        if return_hist:
            hist.append(xs_pred.clone())
            hist = torch.stack(hist)
        if padding > 0:
            xs_pred = xs_pred[:, :-padding]
            hist = hist[:, :, :-padding] if return_hist else None
        if context_guidance is not None:
            xs_pred = xs_pred[::len_cg]
            hist = hist[:, ::len_cg] if return_hist else None

        return xs_pred, hist

    def _get_training_noise_levels(
        self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        batch_size, n_tokens, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_decay":
                if not hasattr(self, "noise_level_probs"):
                    levels = torch.arange(self.timesteps, device=xs.device)
                    gamma = self.cfg.noise_level_kwargs.gamma
                    probs = (1 - levels / self.timesteps) ** (1 / gamma) - (1 - (levels + 1) / self.timesteps) ** (
                        1 / gamma
                    )
                    noise_level_probs = probs / probs.sum()

                noise_levels = torch.multinomial(noise_level_probs, batch_size * n_tokens, replacement=True).view(
                    batch_size, n_tokens
                )

            case "random_all":  # entirely random noise levels
                noise_levels = torch.randint(
                    0, self.timesteps, (batch_size, n_tokens), device=xs.device
                )
            case "random_uniform":  # uniform random noise levels for each batch
                noise_levels = torch.randint(
                    0, self.timesteps, (batch_size, 1), device=xs.device
                ).repeat(1, n_tokens)
            case (
                "random_curr"
            ):  # only random noise levels for the current frame, 0 for all previous frames
                noise_levels = torch.cat(
                    [
                        torch.zeros(
                            batch_size, 1, dtype=torch.long, device=xs.device
                        ).repeat(1, n_tokens - 1),
                        torch.randint(
                            0, self.timesteps, (batch_size, 1), device=xs.device
                        ),
                    ],
                    1,
                )
        if self.cfg.noiseless_context:
            noise_levels[:, : self.n_context_tokens] = 0

        if masks is not None:
            # for frames that are not available, treat as full noise
            masks = reduce(masks.bool(), "b t fs ... -> b t fs", torch.any)
            noise_levels = torch.where(
                masks.all(-1),
                noise_levels,
                torch.full_like(noise_levels, self.timesteps - 1),
            )

        return noise_levels

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        is_interpolation: bool = False,
    ):
        cfg = self.cfg if not is_interpolation else self.cfg.interpolation
        match cfg.scheduling_matrix:
            case "full_sequence":
                scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)[
                    :, None
                ].repeat(horizon, axis=1)
            case _:
                raise NotImplementedError()

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(
            scheduling_matrix
        )

        return scheduling_matrix

    def _reweight_loss(self, loss, weight=None):
        loss = rearrange(loss, "b t (fs c) ... -> b t fs c ...", fs=self.frame_stack)
        if weight is not None:
            expand_dim = len(loss.shape) - len(weight.shape)
            weight = rearrange(
                weight,
                "... -> ..." + " 1" * expand_dim,
            )
            loss = loss * weight

        return loss.mean()

    def _normalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return (xs - mean) / std

    def _unnormalize_x(self, xs):
        shape = [1] * (xs.ndim - self.data_mean.ndim) + list(self.data_mean.shape)
        mean = self.data_mean.reshape(shape)
        std = self.data_std.reshape(shape)
        return xs * std + mean

    def _unstack_and_unnormalize(self, xs):
        xs = rearrange(xs, "b t (fs c) ... -> b (t fs) c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
