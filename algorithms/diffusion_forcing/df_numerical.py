from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat, reduce
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from pytorch_lightning.utilities import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT


from .df_base import DiffusionForcingBase

class DiffusionForcingNumerical(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.is_conditional = cfg.is_conditional

    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        xs_pred = output["xs_pred"]
        xs = output["xs"]

        # visualize 64 samples
        if self.global_step % self.logging.train_vis_freq == 0:
            n = self.logging.train_vis_samples

            if self.is_conditional:
                conditions_vis = rearrange(batch[1], "t b c -> (t c) b 1").contiguous()
                xs_pred_copy = torch.cat([conditions_vis, xs_pred])
                xs_copy = torch.cat([conditions_vis, xs])
                self._visualize(xs_pred_copy[:, :n], name="training_vis/pred_distribution")
                self._visualize(xs_copy[:, :n], name="training_vis/gt_distribution")
            else:
                self._visualize(xs_pred[:, :n], name="training_vis/pred_distribution")
                self._visualize(xs[:, :n], name="training_vis/gt_distribution")

        return output
    
    def on_validation_epoch_end(self, namespace="validation") -> None:
        loss_stats = []
        for i, out in enumerate(self.validation_step_outputs):
            for rg in self.cfg.diffusion.reconstruction_guidance:
                (_, xs_pred, loss) = out[rg]
                if rg != 0:
                    self._visualize(xs_pred, name=f"{namespace}_vis/distribution_{i}_{rg}")
                else:
                    self._visualize(xs_pred, name=f'{namespace}_vis/distribution_{i}')
                loss_stats.append(loss)
                self.log_dict({f"{namespace}/loss_{rg}": torch.stack(loss_stats).mean()})
                self.validation_step_outputs = []

    def on_test_epoch_end(self, namespace="test") -> None:
        self.on_validation_epoch_end(namespace)
    
    @rank_zero_only
    def _visualize(self, xs, name):
        # xs ~ (t b c)
        xs = xs.detach().cpu().numpy()
        subsample = 4
        xs_start = xs[0, :, 0]
        xs = xs[::subsample, :, 0]
        steps, batch_size = xs.shape

        t = np.linspace(0, 1, steps)
        plt.clf()
        plt.xlim(*self.logging.x_lim)
        plt.ylim(*self.logging.y_lim)
        # cmap = cm.get_cmap("coolwarm")
        for i in range(batch_size):
            color = "b" if xs_start[i] > 0 else "r"
            # color = xs[0, i] 
            plt.plot(t, xs[:, i], c=color, alpha=0.2)

        image_path = f'/tmp/numerical.png'
        plt.savefig(image_path)
        self.logger.experiment.log({name: wandb.Image(image_path)})

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks, *_ = batch

        rg_losses = {}

        for rg in self.cfg.diffusion.reconstruction_guidance:
            if self.is_conditional:
                xs_pred, _= self._sample_sequence(
                    xs.shape[1],
                    xs.shape[0],
                    None,
                    None,
                    conditions,
                    None,
                    rg 
                )
            else:
                xs_pred, _ = self.predict_sequence(
                    xs[: self.n_context_tokens],
                    len(xs),
                    conditions,
                    reconstruction_guidance=rg,
                    compositional=self.is_compositional,
            )

            # FIXME: loss
            loss = self.loss_fn(xs_pred, xs, reduction="none")
            loss = self._reweight_loss(loss, masks).detach().cpu()

            xs_copy = self._unstack_and_unnormalize(xs).detach().cpu()
            xs_pred_copy = self._unstack_and_unnormalize(xs_pred).detach().cpu()

            if self.is_conditional:
                conditions_vis = rearrange(conditions, "t b c -> (t c) b 1").contiguous().detach().cpu()
                xs_copy = torch.cat([conditions_vis, xs_copy])
                xs_pred_copy = torch.cat([conditions_vis, xs_pred_copy])

                rg_losses[rg] = (xs_copy,xs_pred_copy,loss)
            else:
                rg_losses[rg] = (xs, xs_pred, loss)

        self.validation_step_outputs.append(rg_losses)

        return rg_losses