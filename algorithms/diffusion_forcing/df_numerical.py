from typing import Optional, Any
from omegaconf import DictConfig
import numpy as np
from random import random
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat, reduce
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .df_base import DiffusionForcingBase


class DiffusionForcingNumerical(DiffusionForcingBase):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def training_step(self, batch, batch_idx):
        output = super().training_step(batch, batch_idx)
        xs_pred = output["xs_pred"]
        xs = output["xs"]

        # visualize 64 samples
        if self.global_step % self.logging.train_vis_freq == 0:
            n = self.logging.train_vis_samples
            self._visualize(xs_pred[:, :n], name="training_vis/pred_distribution")
            self._visualize(xs[:, :n], name="training_vis/gt_distribution")

        return output

    def on_validation_epoch_end(self, namespace="validation") -> None:
        loss_stats = []
        for i, (_, xs_pred, loss) in enumerate(self.validation_step_outputs):
            self._visualize(xs_pred, name=f"{namespace}_vis/distribution_{i}")
            loss_stats.append(loss)
        self.log_dict({f"{namespace}/loss": torch.stack(loss_stats).mean()})
        self.validation_step_outputs = []

    def _visualize(self, xs, name):
        # xs ~ (t b c)
        xs = xs.detach().cpu().numpy()
        subsample = 4
        xs = xs[::subsample, :, 0]
        steps, batch_size = xs.shape

        t = np.linspace(0, 1, steps)
        plt.clf()
        plt.xlim(*self.logging.x_lim)
        plt.ylim(*self.logging.y_lim)
        cmap = cm.get_cmap("coolwarm")
        for i in range(batch_size):
            # color = "b" if xs[0, i] > 0 else "r"
            color = xs[0, i] + 0.5
            plt.plot(t, xs[:, i], c=cmap(color), alpha=0.2)
        plt.savefig(f"/tmp/numerical.png")
        self.logger.experiment.log({name: wandb.Image(f"/tmp/numerical.png")})
