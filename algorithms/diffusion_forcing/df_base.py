"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Optional, Callable, Tuple, Any
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from lightning.pytorch.utilities.types import STEP_OUTPUT

from algorithms.diffusion_forcing.models import Diffusion, Curriculum
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from .models.diffusion import Diffusion


class DiffusionForcingBase(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.frame_skip = cfg.frame_skip
        self.x_stacked_shape = list(self.x_shape)
        self.x_stacked_shape[0] *= cfg.frame_stack
        self.guidance_scale = cfg.guidance_scale
        self.chunk_size = cfg.chunk_size
        self.external_cond_dim = (
            cfg.external_cond_dim
            * cfg.frame_stack
            * (cfg.frame_skip if cfg.external_cond_stack else 1)
        )
        self.use_causal_mask = cfg.diffusion.use_causal_mask
        self.logging = cfg.logging

        self.is_compositional = cfg.is_compositional
        self.unknown_noise_level_prob = cfg.diffusion.unknown_noise_level_prob

        self.uncertainty_scale = cfg.uncertainty_scale
        self.timesteps = cfg.diffusion.timesteps
        self.sampling_timesteps = cfg.diffusion.sampling_timesteps
        self.clip_noise = cfg.diffusion.clip_noise
        self.stabilization_level = cfg.diffusion.stabilization_level
        self.replacement = cfg.replacement

        cfg.diffusion.cum_snr_decay = cfg.diffusion.cum_snr_decay ** (
            self.frame_stack * cfg.frame_skip
        )

        self.curriculum = (
            Curriculum(cfg.curriculum)
            if cfg.curriculum
            else self._build_static_curriculum(cfg)
        )

        self.validation_multiplier = cfg.validation_multiplier
        self.validation_step_outputs = []

        match cfg.loss:
            case "mse":
                self.loss_fn = nn.functional.mse_loss
            case "l1":
                self.loss_fn = nn.functional.l1_loss

        super().__init__(cfg)

    @staticmethod
    def _build_static_curriculum(cfg: DictConfig) -> Curriculum:
        return Curriculum.static(
            n_tokens=cfg.n_frames // cfg.frame_stack,
            n_context_tokens=cfg.context_frames // cfg.frame_stack,
        )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    def _build_model(self):
        self.diffusion_model = Diffusion(
            cfg=self.cfg.diffusion,
            backbone_cfg=self.cfg.backbone,
            x_shape=self.x_stacked_shape,
            external_cond_dim=self.external_cond_dim,
            curriculum=self.curriculum,
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

    @property
    def is_diffusion_forcing(self) -> bool:
        """
        Whether the model is a diffusion forcing model.
        """
        return self.cfg.noise_level in ["random_all", "random_decay"]

    # ---------------------------------------------------------------------
    # Length-related Properties
    # NOTE: "Frame" and "Token" should be distinguished carefully.
    # "Frame" refers to original unit of data loaded from dataset.
    # "Token" refers to the unit of data processed by the diffusion model.
    # There is no guarantee that n_frames == n_tokens * frame_stack.
    # ---------------------------------------------------------------------

    @property
    def orig_n_frames(self) -> int:
        """
        Expected number of frames from dataset, before padding.
        """
        return round(
            self.cfg.n_frames
            * (1 if self.trainer.training else self.validation_multiplier)
        )

    # @property
    # def padding(self) -> int:
    #     """
    #     Padding that orig_n_frames needs to reach multiples of frame stack.
    #     """
    #     tokens = np.ceil(self.orig_n_frames // self.frame_stack)
    #     return tokens * self.frame_stack - self.orig_n_frames

    @property
    def max_tokens(self) -> int:
        """
        Get the maximum number of tokens for the model.
        """
        return self.curriculum.curr_n_tokens

    @property
    def n_tokens(self) -> int:
        """
        Get the number of tokens to be processed.
        """
        return round(
            self.max_tokens
            * (1 if self.trainer.training else self.validation_multiplier)
        )

    @property
    def n_frames(self) -> int:
        """
        Get the number of frames to be processed.
        """
        return self.n_tokens * self.frame_stack

    @property
    def n_context_tokens(self) -> int:
        """
        Get the number of context tokens for the model.
        """
        return self.curriculum.curr_context_tokens

    @property
    def n_context_frames(self) -> int:
        """
        Get the number of context frames for the model.
        """
        return self.n_context_tokens * self.frame_stack

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
            xs: tensor of shape (n_tokens, batch_size, *stacked_x_shape)
            conditions: optional tensor of shape (n_tokens, batch_size, d)
            masks: tensor of shape (n_tokens, batch_size, frame_stack)
        """

        # optional curriculum pretrains with shorter sequences first (e.g. image)
        if self.trainer.training:
            self.curriculum(self.global_step)

        xs = self._normalize_x(batch["xs"])
        conditions = None

        n_frames = xs.shape[1]

        # if self.padding > 0:
        #     raise ValueError("n_frames must be a multiple of frame_stack.")

        #     if self.validation_multiplier > 1:
        #         raise ValueError(
        #             "Validation multiplier > 1 is not compatible with padding."
        #         )
        #     n_tokens = (n_frames + self.padding) // self.frame_stack
        #     if n_tokens != self.n_tokens:
        #         raise ValueError("Padding is only supported without curriculum.")

        #     pad = torch.zeros_like(xs[:, : self.padding])
        #     xs = torch.cat([xs, pad], 1)

        if self.external_cond_dim:
            if conditions.shape[-1] != self.external_cond_dim:
                raise ValueError(
                    f"Expected external condition dim {self.external_cond_dim}, got {conditions.shape[-1]}."
                )
            # if self.padding > 0:
            #     conditions = nn.functional.pad(conditions, (0, 0, 0, self.padding))

            conditions = rearrange(
                batch["conds"],
                "b (n t fs) d -> t (b n) (fs d)",
                t=self.n_tokens,
                fs=self.frame_stack,
            ).contiguous()

        xs = rearrange(
            xs,
            "b (n t fs) c ... -> t (b n) (fs c) ...",
            fs=self.frame_stack,
            t=self.n_tokens,
        ).contiguous()

        if "masks" in batch:
            masks = rearrange(
                batch["masks"],
                "b (n t fs) ... -> t (b n) fs ...",
                fs=self.frame_stack,
                t=self.n_tokens,
            ).contiguous()
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
        n_tokens, batch_size, *_ = xs.shape

        if self.cfg.noise_level == "random_curr":  # for teacher forcing
            raise NotImplementedError
            # randomly cut the sequence and predict the last frame
            t = torch.randint(1, n_tokens + 1, (1,)).item()
            noise_levels = torch.cat(
                [
                    self._generate_noise_levels(xs[:t], masks),
                    torch.zeros(
                        n_tokens - t, batch_size, dtype=torch.long, device=xs.device
                    ),
                ],
                0,
            )
            xs_pred, loss = self.diffusion_model(
                xs, conditions, noise_levels=noise_levels
            )
            # only consider the loss of the chosen frame
            loss = loss[t - 1 : t]
            masks = masks[(t - 1) * self.frame_stack : t * self.frame_stack]
            xs, xs_pred = xs[:t], xs_pred[:t]

        else:
            xs_pred, loss = self.diffusion_model(
                xs,
                conditions,
                noise_levels=self._generate_noise_levels(xs, masks),
            )

        if self.cfg.noiseless_context:
            masks[: self.n_context_tokens * self.frame_stack] = False

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
            xs[: self.n_context_tokens],
            len(xs),
            conditions,
            reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
            compositional=self.is_compositional,
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
        compositional: bool = False,
        return_hist: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict a sequence using the model, using context tokens at the front.
        If length is bigger than self.max_tokens, will use sliding window sampling.

        Returns:
            xs_pred: shape (length, batch_size, *self.x_stacked_shape)
            hist: if True, history of xs_pred (m, length, batch_size, *self.x_stacked_shape) on cpu. None otherwise.
        """
        if length is None:
            length = self.max_tokens
        xs_pred = context
        x_shape = self.x_stacked_shape
        batch_size = context.shape[1]
        curr_token = xs_pred.shape[0]
        gt_len = curr_token

        # if self.n_context_tokens < curr_token:
        #     raise ValueError("self.n_context_tokens should be < length of context.")
        if conditions is not None:
            if len(conditions) < length:
                raise ValueError(
                    f"conditions length is expected to be at least the length {length}."
                )
            # if not self.use_causal_mask and self.chunk_size >= 0:
            #     raise ValueError(
            #         "Non-causal models that require external conditions must have self.chunk_size = -1 for sampling, aka sampling maximum sequence length jointly."
            #     )

        hist = None
        while curr_token < length:
            if hist is not None:
                raise ValueError(
                    "return_hist is not supported if using sliding window."
                )

            c = min(self.n_context_tokens, curr_token)
            if conditions is not None and not self.use_causal_mask:
                c = max(c, curr_token - (len(conditions) - self.max_tokens))

            h = min(length - curr_token, self.max_tokens - c)
            h = min(h, self.chunk_size) if self.chunk_size > 0 else h
            l = c + h
            pad = torch.zeros((h, batch_size, *x_shape)).to(xs_pred.device)
            context = torch.cat([xs_pred[-c:], pad])
            context_mask = torch.ones((c, batch_size, *x_shape)).to(context.device)
            generated_len = curr_token - max(curr_token - c, gt_len)
            context_mask[-generated_len:] = -1
            context_mask = torch.cat([context_mask.bool(), pad.bool()])

            cond_len = l if self.use_causal_mask else self.max_tokens
            cond_slice = None
            if conditions is not None:
                cond_slice = conditions[curr_token - c : curr_token - c + cond_len]

            new_pred, hist = self._sample_sequence(
                batch_size,
                length=l,
                context=context,
                context_mask=context_mask,
                conditions=cond_slice,
                guidance_fn=guidance_fn,
                reconstruction_guidance=reconstruction_guidance,
                compositional=compositional,
                return_hist=return_hist,
            )
            xs_pred = torch.cat([xs_pred, new_pred[-h:]])
            curr_token = xs_pred.shape[0]
        return xs_pred, hist

    def _sample_sequence(
        self,
        batch_size: int,
        length: Optional[int] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
        reconstruction_guidance: float = 0.0,
        compositional: bool = False,
        return_hist: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        The unified sampling method, with length up to maximum token size.
        context of same length can be provided with a mask to achieve conditioning.

        Args:
            batch_size: batch size
            length: number of frames in sampled sequence, if None, fall back to context t and then fall back to self.max_tokens
            context: shape (length, batch_size, *self.x_stacked_shape)
            context_mask: same shape as context, -1 for generated context, 1 for gt context, 0 for no context
            conditions: conditions external conditions for sampling
            guidance_fn: guidance function for sampling
            reconstruction_guidance: the scale of reconstruction guidance for True entries in context_mask
            compositional: if True, sample with context frames as unknown noise level tokens
            return_hist: if True, return all steps of the sampling process
        Returns:
            xs_pred: shape (length, batch_size, *self.x_stacked_shape)
            hist: if True, history of xs_pred (m, length, batch_size, *self.x_stacked_shape) on cpu. None otherwise.
        """
        if length is None:
            length = self.max_tokens if context is None else context.shape[0]

        x_shape = self.x_stacked_shape

        if length > self.max_tokens:
            raise ValueError(
                f"length is expected to <={self.max_tokens}, got {length}."
            )

        if context is not None:
            if context_mask is None:
                raise ValueError("context_mask must be provided if context is given.")
            if context.shape[1] != batch_size:
                raise ValueError(
                    f"context batch size is expected to be {batch_size} but got {context.shape[1]}."
                )
            if context.shape[0] != length:
                raise ValueError(
                    f"context length is expected to be {length} but got {context.shape[0]}."
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

        if conditions is not None:
            if self.use_causal_mask and len(conditions) != length:
                raise ValueError(
                    f"for causal models, conditions length is expected to be {length}, got {len(conditions)}."
                )
            elif not self.use_causal_mask and len(conditions) != self.max_tokens:
                raise ValueError(
                    f"for noncausal models, conditions length is expected to be {self.max_tokens}, got {len(conditions)}."
                )

        replacement_mask = context_mask.bool()  # both -1 and 1 map to True

        # compositional = compositional and self.unknown_noise_level_prob > 0
        h = length if self.use_causal_mask else self.max_tokens
        padding = 0
        xs_pred = torch.randn((h, batch_size, *x_shape), device=self.device)
        xs_pred = torch.clamp(xs_pred, -self.clip_noise, self.clip_noise)
        if context is None:
            context = torch.zeros_like(xs_pred)
            replacement_mask = torch.zeros_like(xs_pred, dtype=torch.bool)
        elif not self.use_causal_mask:
            padding = self.max_tokens - length
            context_pad = torch.zeros((padding, batch_size, *x_shape))
            context_pad = context_pad.to(device=self.device)
            replacement_mask_pad = torch.zeros_like(context_pad, dtype=torch.bool)
            context = torch.cat([context, context_pad])
            replacement_mask = torch.cat([replacement_mask, replacement_mask_pad])
            context_mask = torch.cat([context_mask, replacement_mask_pad.long()])

        # replace xs_pred's context frames with context
        xs_pred = torch.where(replacement_mask, context, xs_pred)

        # generate scheduling matrix
        scheduling_matrix = self._generate_scheduling_matrix(
            h - padding,
            padding,
            is_interpolation=False,
            mask=reduce(replacement_mask, "t ... -> t", torch.all)[: h - padding],
        )
        scheduling_matrix = scheduling_matrix.to(self.device)

        # fill context frame noise levels in scheduling matrix
        scheduling_matrix = repeat(scheduling_matrix, "m t -> m t b", b=batch_size)
        stable_schedule = torch.full_like(
            scheduling_matrix[0], self.stabilization_level - 1
        )
        gt_schedule = torch.full_like(scheduling_matrix, -1)
        reduced_gt_mask = reduce(context_mask == 1, "t b ... -> t b", torch.all)
        gt_mask = repeat(reduced_gt_mask, "t b -> m t b", m=scheduling_matrix.shape[0])
        stable_mask = torch.zeros_like(reduced_gt_mask)
        reduced_replacement_mask = reduce(replacement_mask, "t b ... -> t b", torch.all)
        scheduling_matrix = torch.where(gt_mask, gt_schedule, scheduling_matrix)

        # prune scheduling matrix
        diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
        skip = torch.argmax((~reduce(diff == 0, "m t b -> m", torch.all)).float())
        scheduling_matrix = scheduling_matrix[skip:]

        hist = []
        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = scheduling_matrix[m]
            to_noise_levels = scheduling_matrix[m + 1]
            if self.is_diffusion_forcing and self.stabilization_level > 0:
                stable_mask = ~reduced_gt_mask & (from_noise_levels == -1)
                # stabilize all non-GT frames that are fully diffused
                from_noise_levels = torch.where(
                    stable_mask,
                    stable_schedule,
                    from_noise_levels,
                )
                to_noise_levels = torch.where(
                    stable_mask,
                    stable_schedule,
                    to_noise_levels,
                )

            # optionally mask out noise level of context frames if compositional
            # (noise level will be replaced by unknown token if unknown_noise_level_prob > 0)
            # if compositional:
            #     ukn_noise_mask = reduced_replacement_mask.clone()
            # else:
            #     ukn_noise_mask = torch.zeros_like(gt_mask[m])

            # new compositionality scheme of estimating noise from gaussian
            ukn_noise_mask = torch.zeros_like(gt_mask[m])
            if compositional:
                optimal_noise_level = self.diffusion_model.estimate_noise_level(context)
                from_noise_levels = torch.where(
                    context_mask, optimal_noise_level, from_noise_levels
                )
                to_noise_levels = torch.where(
                    context_mask, optimal_noise_level, to_noise_levels
                )

            # replace with scaled xs_pred if:
            # 1. context frames with noise levels >= 0 (replacement technique for Full Sequence)
            # 2. generated frames that are being stabilized (for Diffusion Forcing)
            if self.replacement is not None:
                prev_xs_pred = xs_pred
                scaled_xs_pred = self.diffusion_model.q_sample(
                    xs_pred,
                    from_noise_levels,
                    noise=(
                        None
                        if self.replacement == "noisy_scale"
                        else torch.zeros_like(xs_pred)
                    ),
                )
                scale_mask = rearrange(
                    (reduced_replacement_mask & (from_noise_levels >= 0)) | stable_mask,
                    "t b -> t b" + " 1" * (context.ndim - 2),
                )
                xs_pred = torch.where(scale_mask, scaled_xs_pred, xs_pred)

            # record value if return history
            if return_hist:
                hist.append(xs_pred.detach())

            # add reconstruction_guidance
            if reconstruction_guidance > 0 and context is not None:

                def composed_guidance_fn(
                    xk: torch.Tensor, pred_x0: torch.Tensor, alpha_cumprod: torch.Tensor
                ) -> torch.Tensor:

                    loss = (
                        self.loss_fn(pred_x0, context, reduction="none")
                        * alpha_cumprod.sqrt()
                    )
                    # scale inversely proportional to the number of context frames
                    loss = torch.sum(
                        loss
                        * replacement_mask
                        / replacement_mask.sum(dim=0, keepdim=True).clamp(min=1),
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
                ukn_noise_mask,
                guidance_fn=composed_guidance_fn,
            )

            if self.replacement is not None:
                xs_pred = torch.where(scale_mask, prev_xs_pred, xs_pred)

        if padding > 0:
            xs_pred = xs_pred[:-padding]

        hist = torch.stack(hist) if return_hist else None

        return xs_pred, hist

    def _generate_noise_levels(
        self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate noise levels for training.
        """
        n_tokens, batch_size, *_ = xs.shape
        match self.cfg.noise_level:
            case "random_decay":
                if not hasattr(self, "noise_level_probs"):
                    levels = torch.arange(self.timesteps, device=xs.device)
                    gamma = self.cfg.noise_level_kwargs.gamma
                    probs = (1 - levels / self.timesteps) ** (1 / gamma) - (
                        1 - (levels + 1) / self.timesteps
                    ) ** (1 / gamma)
                    self.noise_level_probs = probs / probs.sum()

                noise_levels = torch.multinomial(
                    self.noise_level_probs, n_tokens * batch_size, replacement=True
                ).reshape(n_tokens, batch_size)

            case "random_all":  # entirely random noise levels
                noise_levels = torch.randint(
                    0, self.timesteps, (n_tokens, batch_size), device=xs.device
                )
            case "random_uniform":  # uniform random noise levels for each batch
                noise_levels = torch.randint(
                    0, self.timesteps, (1, batch_size), device=xs.device
                ).repeat(n_tokens, 1)
            case (
                "random_curr"
            ):  # only random noise levels for the current frame, 0 for all previous frames
                noise_levels = torch.cat(
                    [
                        torch.zeros(
                            1, batch_size, dtype=torch.long, device=xs.device
                        ).repeat(n_tokens - 1, 1),
                        torch.randint(
                            0, self.timesteps, (1, batch_size), device=xs.device
                        ),
                    ],
                    0,
                )
        if self.cfg.noiseless_context:
            noise_levels[: self.n_context_tokens] = 0

        if masks is not None:
            # for frames that are not available, treat as full noise
            masks = reduce(masks.bool(), "t b fs ... -> t b fs", torch.any)
            noise_levels = torch.where(
                masks.all(-1),
                noise_levels,
                torch.full_like(noise_levels, self.timesteps - 1),
            )

        return noise_levels

    def _generate_scheduling_matrix(
        self,
        horizon: int,
        padding: int = 0,
        is_interpolation: bool = False,
        mask: Optional[torch.Tensor] = None,
    ):
        cfg = self.cfg if not is_interpolation else self.cfg.interpolation
        match cfg.scheduling_matrix:
            case "pyramid":
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon, cfg.uncertainty_scale
                )
            case "full_sequence" | "full_pyramid":
                scheduling_matrix = np.arange(self.sampling_timesteps, -1, -1)[
                    :, None
                ].repeat(horizon, axis=1)
            case "autoregressive":
                scheduling_matrix = self._generate_pyramid_scheduling_matrix(
                    horizon, self.sampling_timesteps
                )
            case "trapezoid":
                scheduling_matrix = self._generate_trapezoid_scheduling_matrix(
                    horizon, self.uncertainty_scale
                )

        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()

        # convert ddim index to ddpm noise level
        if cfg.scheduling_matrix == "full_pyramid":
            if mask is None:
                raise ValueError("mask must be provided for full pyramid scheduling.")
            non_context_frames = torch.nonzero(~mask)
            first_frame = non_context_frames[0].item()
            last_frame = non_context_frames[-1].item()
            scheduling_matrix = self.diffusion_model.ddim_idx_to_pyramid_noise_level(
                scheduling_matrix,
                first_frame,
                last_frame,
                self.cfg.start_exponent,
                self.cfg.end_exponent,
            )
        else:
            scheduling_matrix = self.diffusion_model.ddim_idx_to_noise_level(
                scheduling_matrix
            )

        # paded entries are labeled as pure noise
        scheduling_matrix = nn.functional.pad(
            scheduling_matrix, (0, padding, 0, 0), value=self.timesteps - 1
        )

        return scheduling_matrix

    def _generate_pyramid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float
    ):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = (
                    self.sampling_timesteps + int(t * uncertainty_scale) - m
                )

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(
        self, horizon: int, uncertainty_scale: float
    ):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = (
                    self.sampling_timesteps + int(t * uncertainty_scale) - m
                )
                scheduling_matrix[m, -t] = (
                    self.sampling_timesteps + int(t * uncertainty_scale) - m
                )

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _reweight_loss(self, loss, weight=None):
        loss = rearrange(loss, "t b (fs c) ... -> t b fs c ...", fs=self.frame_stack)
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
        xs = rearrange(xs, "t b (fs c) ... -> (t fs) b c ...", fs=self.frame_stack)
        return self._unnormalize_x(xs)
