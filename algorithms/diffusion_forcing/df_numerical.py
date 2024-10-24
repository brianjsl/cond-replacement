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
        self.is_conditional = cfg.conditional

    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        xs_pred = output["xs_pred"]
        xs = output["xs"]

        # visualize 64 samples
        if self.global_step % self.logging.train_vis_freq == 0:
            n = self.logging.train_vis_samples

            if self.is_conditional and self.logger is not None:
                conditions_vis = rearrange(batch[1], "b t c -> b (t c) 1")
                xs_pred_copy = torch.cat([conditions_vis, xs_pred], dim = 1)
                xs_copy = torch.cat([conditions_vis, xs], dim=1)
                self._visualize(xs_pred_copy[:n], name="training_vis/pred_distribution")
                self._visualize(xs_copy[:n], name="training_vis/gt_distribution")
            else:
                self._visualize(xs_pred[:n], name="training_vis/pred_distribution")
                self._visualize(xs[:n], name="training_vis/gt_distribution")

        return output
    
    def on_validation_epoch_end(self, namespace="validation") -> None:
        loss_stats = []
        for out in self.validation_step_outputs:
            (_, xs_pred, loss) = out

            type_name = ''
            if self.is_conditional:
                type_name += '_conditional'
            elif self.cfg.diffusion.reconstruction_guidance != 0.0:
                type_name += f'_rg_{self.cfg.diffusion.reconstruction_guidance}'
            else:
                type_name += '_no_rg'

            self._visualize(xs_pred, name=f'{namespace}_vis/distribution'+ type_name)
            loss_stats.append(loss)
            self.log_dict({f"{namespace}/loss"+type_name: torch.stack(loss_stats).mean()})
        self.validation_step_outputs = []

    def on_test_epoch_end(self, namespace="test") -> None:
        self.on_validation_epoch_end(namespace)
    
    @rank_zero_only
    def _visualize(self, xs, name):
        # xs ~ (b t c)
        xs = xs.detach().cpu().numpy()
        subsample = 4
        xs_start = xs[:,0, 0]
        xs = xs[:,::subsample,0]
        batch_size, steps = xs.shape

        t = np.linspace(0, 1, steps)
        plt.clf()
        plt.xlim(*self.logging.x_lim)
        # plt.ylim(*self.logging.y_lim)
        # cmap = cm.get_cmap("coolwarm")
        for i in range(batch_size):
            color = "b" if xs_start[i] > 0 else "r"
            # color = xs[0, i] 
            plt.plot(t, xs[i, :], c=color, alpha=0.2)

        image_path = f'/tmp/numerical.png'
        plt.savefig(image_path)
        self.logger.experiment.log({name: wandb.Image(image_path)})

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, namespace="validation") -> STEP_OUTPUT:
        xs, conditions, masks, *_ = batch

        if self.is_conditional:
            xs_pred, _ = self._sample_sequence(
                xs.shape[0], 
                xs.shape[1],
                None,
                None,
                None,
                conditions,
                None,
                self.cfg.diffusion.reconstruction_guidance
            )
        else:
            xs_pred, _ = self.predict_sequence(
                xs[:, : self.n_context_tokens],
                xs.shape[1],
                conditions,
                reconstruction_guidance=self.cfg.diffusion.reconstruction_guidance,
                context_guidance=None,
                rg_monte_carlo=self.cfg.diffusion.rg_monte_carlo,
                monte_carlo_n=self.cfg.diffusion.monte_carlo_n
            )

        # FIXME: loss
        loss = self.loss_fn(xs_pred, xs, reduction="none")
        loss = self._reweight_loss(loss, masks).detach().cpu()


        if self.is_conditional:
            xs_copy = self._unstack_and_unnormalize(xs).detach().cpu()
            xs_pred_copy = self._unstack_and_unnormalize(xs_pred).detach().cpu()
            conditions_vis = rearrange(conditions, "b t c -> b (t c) 1").detach().cpu()
            xs_copy = torch.cat([conditions_vis, xs_copy], dim=1)
            xs_pred_copy = torch.cat([conditions_vis, xs_pred_copy], dim=1)

            self.validation_step_outputs.append((xs_copy,xs_pred_copy,loss))
        else:
            xs = self._unstack_and_unnormalize(xs).detach().cpu()
            xs_pred = self._unstack_and_unnormalize(xs_pred).detach().cpu()
            self.validation_step_outputs.append((xs, xs_pred, loss))
            
        return loss
