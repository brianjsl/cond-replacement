from typing import List, Literal, Dict
import torch
from torch import Tensor
from torch.nn import ModuleDict
from einops import rearrange
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print
from . import (
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
    FrechetVideoDistance,
    MeanSquaredError,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    UniversalImageQualityIndex,
)


class VideoMetric(ModuleDict):
    """
    A class that wraps all video metrics.
    """

    def __init__(
        self,
        metric_types: List[
            Literal["lpips", "fid", "fvd", "mse", "ssim", "psnr", "uiqi"]
        ],
    ):
        modules = {}
        for metric_type in metric_types:
            match metric_type:
                case "lpips":
                    module = LearnedPerceptualImagePatchSimilarity()  # (-1, 1)
                case "fid":
                    module = FrechetInceptionDistance(
                        feature=64, normalize=True
                    )  # (0, 1)
                case "fvd":
                    module = FrechetVideoDistance()  # (-1, 1)
                case "mse":
                    module = MeanSquaredError()  # (0, 1)
                case "ssim":
                    module = StructuralSimilarityIndexMeasure(data_range=1.0)  # (0, 1)
                case "psnr":
                    module = PeakSignalNoiseRatio(data_range=1.0)  # (0, 1)
                case "uiqi":
                    module = UniversalImageQualityIndex()  # (0, 1)
            modules[metric_type] = module

        super().__init__(modules)

    def forward(self, preds: Tensor, target: Tensor):
        # replace all NaNs with 0
        preds = torch.nan_to_num(preds, nan=0.0)
        target = torch.nan_to_num(target, nan=0.0)
        # clamp to [0, 1]
        preds = torch.clamp(preds, 0.0, 1.0).to(torch.float32)
        target = torch.clamp(target, 0.0, 1.0).to(torch.float32)

        if "fvd" in self:
            if (preds.shape[0] < 9) or (target.shape[0] < 9):
                rank_zero_print(
                    cyan("FVD requires at least 9 frames, skipping FVD computation.")
                )
            else:
                self["fvd"].update(target * 2.0 - 1.0, real=True)
                self["fvd"].update(preds * 2.0 - 1.0, real=False)

        preds = rearrange(preds, "t b c h w -> (t b) c h w")
        target = rearrange(target, "t b c h w -> (t b) c h w")

        for metric_type, module in self.items():
            if metric_type in ["mse", "ssim", "psnr", "uiqi"]:
                if metric_type == "uiqi":
                    # limit buffer size to 10 GB for UIQI
                    buffer_size = (
                        sum(
                            pred.element_size() * pred.nelement()
                            for pred in module.preds
                        )
                        / 1024**3
                    )
                    if buffer_size > 10.0:
                        rank_zero_print(
                            cyan(
                                f"Universal Image Quality Index buffer size is {buffer_size:.2f} GB, skipping UIQI computation."
                            )
                        )
                        continue

                module.update(preds, target)
            elif metric_type == "lpips":
                module.update(preds * 2.0 - 1.0, target * 2.0 - 1.0)
            elif metric_type == "fid":
                module.update(target, real=True)
                module.update(preds, real=False)

    def log(self, prefix: str):
        return {
            f"{prefix}/{metric_type}": module
            for metric_type, module in self.items()
            if metric_type != "fvd" or module.real_features_num_samples.item() > 0
        }

    def reset(self):
        for module in self.values():
            module.reset()
