from typing import Any, Dict, Tuple
from omegaconf import DictConfig, open_dict
import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from algorithms.common.utils import warmup
from algorithms.common.metrics.video_metric import VideoMetric
from algorithms.vae.common.losses import LPIPSWithDiscriminator3D
from utils.distributed_utils import is_rank_zero
from utils.logging_utils import log_video
from .model import CausalVideoVAE


class TrainableCausalVideoVAE(BasePytorchAlgo):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.lr = cfg.lr
        self.disc_start = cfg.loss.disc_start
        self.warmup_steps = cfg.training.warmup_steps
        self.gradient_clip_val = cfg.training.gradient_clip_val
        self.num_logged_videos = 0
        super().__init__(cfg)

    def _build_model(self):
        with open_dict(self.cfg):
            for key, value in self.cfg.model.items():
                if isinstance(value, list):
                    self.cfg.model[key] = tuple(value)
        self.vae = CausalVideoVAE(**self.cfg.model)
        self.loss = LPIPSWithDiscriminator3D(**self.cfg.loss)
        self.metrics = VideoMetric(self.cfg.logging.metrics)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # save model config to enable loading the model from checkpoint only
        checkpoint["model_cfg"] = self.cfg.model

    def _load_ema_weights_to_state_dict(self, checkpoint: dict) -> None:
        vae_ema_weights = checkpoint["optimizer_states"][0]["ema"]
        vae_parameter_keys = ["vae." + k for k, _ in self.vae.named_parameters()]
        assert len(vae_ema_weights) == len(vae_parameter_keys)
        for key, weight in zip(vae_parameter_keys, vae_ema_weights):
            checkpoint["state_dict"][key] = weight

    def configure_optimizers(self) -> OptimizerLRScheduler:
        self.automatic_optimization = False
        optimizer_vae = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.lr,
            betas=self.cfg.training.optimizer_beta,
        )
        optimizer_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=self.lr,
            betas=self.cfg.training.optimizer_beta,
        )
        return [optimizer_vae, optimizer_disc], []

    def on_after_batch_transfer(
        self, batch: Dict[str, torch.Tensor], dataloader_idx: int = 0
    ) -> torch.Tensor:
        x = batch["videos"]
        return self._rearrange_and_normalize(x)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, namespace: str = "training"
    ):
        is_training = namespace == "training"
        recons, posterior = self.vae(batch)

        if is_training:
            optimizer_vae, optimizer_disc = self.optimizers()
            warmup_info = self._compute_warmup()

        # Optimize VAE
        vae_loss, vae_loss_dict = self.loss(
            inputs=batch,
            reconstructions=recons,
            posteriors=posterior,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.vae.get_last_layer(),
            split=f"{namespace}_vae",
        )
        if is_training:
            self._optimizer_step(optimizer_vae, vae_loss, warmup_info)
        self._log_losses(f"{namespace}_vae", vae_loss, vae_loss_dict, is_training)

        # Optimize Discriminator
        disc_loss, disc_loss_dict = self.loss(
            inputs=batch,
            reconstructions=recons,
            posteriors=posterior,
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=None,
            split=f"{namespace}_disc",
        )
        if is_training:
            self._optimizer_step(optimizer_disc, disc_loss, warmup_info)
        self._log_losses(f"{namespace}_disc", disc_loss, disc_loss_dict, is_training)

        return {
            "gts": self._rearrange_and_unnormalize(batch),
            "recons": self._rearrange_and_unnormalize(recons),
        }

    def on_validation_epoch_start(self) -> None:
        self.num_logged_videos = 0

    def on_validation_epoch_end(self, namespace: str = "validation") -> None:
        # Log metrics
        self.log_dict(
            self.metrics.log(namespace),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end(namespace="test")

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, namespace: str = "validation"
    ) -> STEP_OUTPUT:
        output_dict = self.training_step(batch, batch_idx, namespace)

        # Update metrics
        gts, recons = output_dict["gts"], output_dict["recons"]
        self.metrics(recons, gts)

        # Log ground truth and reconstruction videos
        gts, recons = self.gather_data((gts, recons))
        if not (
            is_rank_zero
            and self.logger
            and self.num_logged_videos < self.cfg.logging.max_num_videos
        ):
            return
        num_videos_to_log = min(
            self.cfg.logging.max_num_videos - self.num_logged_videos,
            gts.shape[1],
        )
        gts, recons = map(
            lambda x: x[:, :num_videos_to_log],
            (gts, recons),
        )
        log_video(
            recons,
            gts,
            step=None if namespace == "test" else self.global_step,
            namespace=f"{namespace}_vis",
            logger=self.logger.experiment,
            indent=self.num_logged_videos,
        )
        self.num_logged_videos += num_videos_to_log

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> STEP_OUTPUT:
        return self.validation_step(batch, batch_idx, namespace="test")

    def _log_losses(
        self,
        namespace: str,
        loss: torch.Tensor,
        loss_dict: Dict[str, torch.Tensor],
        on_step: bool = True,
    ):
        if self.global_step % self.cfg.logging.loss_freq > 1:
            return
        loss_dict = {
            k: v.to(self.device) for k, v in loss_dict.items()
        }  # to enable gathering across devices
        self.log(
            f"{namespace}/loss",
            loss,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=True,
            sync_dist=True,
        )
        self.log_dict(
            loss_dict,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            sync_dist=True,
        )

    def _optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
        warmup_info: Tuple[bool, float],
    ) -> None:
        should_warmup, lr_scale = warmup_info
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val)
        if should_warmup:
            optimizer = warmup(optimizer, self.lr, lr_scale)
        optimizer.step()

    def _compute_warmup(self) -> Tuple[bool, float]:
        should_warmup, lr_scale = False, 1.0
        if self.global_step < self.warmup_steps:
            should_warmup = True
            lr_scale = float(self.global_step + 1) / self.warmup_steps
        elif (
            self.global_step >= self.disc_start - 1
            and self.global_step < self.disc_start + self.warmup_steps
        ):
            should_warmup = True
            lr_scale = float(self.global_step - self.disc_start + 1) / self.warmup_steps
        return should_warmup, min(lr_scale, 1.0)

    def _rearrange_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t c h w -> b c t h w")
        return 2.0 * x - 1.0

    def _rearrange_and_unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        x = 0.5 * x + 0.5
        return rearrange(x, "b c t h w -> t b c h w")
