from datasets.numerical.function1d import DoubleConeDataset
from omegaconf import DictConfig
import yaml
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

if __name__ == '__main__':
    cfg = MagicMock()
    cfg.observation_shape = [1]
    cfg.n_frames = 200
    cfg.frame_skip= 1
    cfg.context_length= 100 #length of context (usually half of n_frames)
    cfg.data_mean= 0.0
    cfg.data_std= 0.5
    cfg.bazier_degree= 15 #number of points used-1
    cfg.purturbation= 0.05
    cfg.spike_multiplier= 20
    cfg.external_cond_dim= 1
    dataset = DoubleConeDataset(cfg, 'training')
    print(dataset[0]['conditions'].shape)
