from .models.diffusion import Diffusion
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
from omegaconf import DictConfig

class DiffNumerical(BasePytorchAlgo):
    def __init__(self, cfg: DictConfig):
        self.x_shape = cfg.x_shape
        self.frame_stack = cfg.frame_stack
        self.frame_skip = cfg.frame_skip