from typing import Any
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import FrechetInceptionDistance
from utils.torch_utils import freeze_model
from .fvd import open_url


class FrechetVideoDistance(FrechetInceptionDistance):
    """
    Calculates Fréchet video distance (FVD) to quantify the similarity between two video distributions.
    Adapted from https://github.com/cvpr2022-stylegan-v/stylegan-v to comply with torchmetrics & PyTorch Lightning.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(self, reset_real_features: bool = True, **kwargs: Any) -> None:
        # pylint: disable=non-parent-init-called
        Metric.__init__(self, **kwargs)

        with open_url(
            "https://github.com/kwsong0113/FVD/raw/main/i3d_torchscript.pt",
            # comes from
            # https://github.com/JunyaoHu/common_metrics_on_video_quality/raw/main/fvd/styleganv/i3d_torchscript.pt
            verbose=False,
        ) as f:
            detector = torch.jit.load(f)
        detector.eval()
        freeze_model(detector)

        def fixed_eval_train(self, mode: bool):
            # pylint: disable=bad-super-call
            return super(torch.jit.ScriptModule, self).train(False)

        # pylint: disable=no-value-for-parameter
        detector.train = fixed_eval_train.__get__(detector, torch.jit.ScriptModule)
        self.detector = detector

        self.detector_kwargs: dict = dict(
            rescale=False, resize=True, return_features=True
        )

        self.reset_real_features = reset_real_features
        num_features = 400

        mx_nb_feets = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_nb_feets).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, videos: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        videos = torch.clamp(videos, -1.0, 1.0).permute(1, 2, 0, 3, 4).contiguous()
        features = self.detector(videos, **self.detector_kwargs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.size(0)
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.size(0)
