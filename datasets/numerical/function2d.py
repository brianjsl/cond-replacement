from abc import ABC, abstractmethod
import torch
from omegaconf import DictConfig
from typing import Literal, Dict
import torch

class Function2D(ABC, torch.utils.data.Dataset):
    '''
    Abstract class for 2D function. 
    '''

    SPLIT = Literal['train', 'val', 'test']
    
    def __init__(self, cfg:DictConfig, split: SPLIT):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.n_points = cfg.n_points
        torch.manual_seed(cfg.random_seed)
    
    def __len__(self):
        return 100000000
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

class LinearNoisy(Function2D):
    '''
    Linear with Additive Noise.
    '''
    
    def __init__(self, cfg:DictConfig, split: Function2D.SPLIT, 
                 *args, **kwargs):
        super().__init__(cfg, split)
        self.a = cfg.a
        self.b = cfg.b
        self.x_dist = cfg.x_dist
        self.sigma = cfg.sigma

        self.x_cond = None
        if split == 'val':
            self.x_cond = cfg.val_x_cond
        elif split == 'test':
            self.x_cond = cfg.test_x_cond

        assert self.x_dist in ['beta']
        match self.x_dist:
            case 'beta':
                self.x_dist = torch.distributions.Beta(cfg.x_conc1, cfg.x_conc2)
        self.noise_fn = cfg.noise_fn
        assert self.noise_fn in ['normal', 'beta']
        match self.noise_fn:
            case 'normal':
                self.noise_fn = torch.distributions.Normal(0, 1)
            case 'beta':
                self.noise_fn = torch.distributions.Beta(cfg.y_conc1,cfg.y_conc2)

        if self.x_cond:
            self.data_points= dict(xs=self.x_cond)
        else: 
            x = self.x_dist.sample(sample_shape=[self.n_points])
            y = self.a * x + self.b + torch.randn_like(x) * self.sigma * self.noise_fn.sample(x.shape)
            output_dict = dict(xs=x, ys=y)
            self.data_points =  output_dict
        
    def __getitem__(self, idx):
        return dict(x=self.data_points['xs'][idx], y=self.data_points['ys'][idx])
    
    def __call__(self):
        return self.data_points
        