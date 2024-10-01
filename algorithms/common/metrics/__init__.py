from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    UniversalImageQualityIndex,
)
from .fvd import DeprecatedFrechetVideoDistance
from .fvd_metric import FrechetVideoDistance
from .video_metric import VideoMetric
