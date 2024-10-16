from abc import ABC, abstractmethod
from typing import Union, Dict
from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import h5py
import urllib
import os
from bezier.curve import Curve
from typing import Literal


class Function1DDataset(ABC, torch.utils.data.Dataset):
    SPLIT = Literal['train', 'val', 'test']
    
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.conditional = cfg.conditional
        self.n_frames = cfg.n_frames
        self.context_length = cfg.context_length
        np.random.seed(0)

    @abstractmethod
    def base_function(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        raise NotImplementedError

    def __len__(self):
        return 2048000

    @abstractmethod
    def __getitem__(self, idx) -> Dict:
        raise NotImplementedError

class DoubleConeDataset(Function1DDataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__(cfg, split)
        self.n_nodes = cfg.bazier_degree + 1
        self.purturbation = cfg.purturbation
        self.spike_multiplier = cfg.spike_multiplier

    def base_function(self, t: np.ndarray) -> np.ndarray | float:
        k = np.random.random() * 2 - 1
        y = k * (t - 0.5)
        return y

    def purturb_function(self, t, purturbation: Union[np.ndarray, float] = 0.05):
        if isinstance(purturbation, float):
            purturbation = np.ones(self.n_nodes) * purturbation
        else:
            assert len(purturbation) == self.n_nodes
        control_t = np.linspace(0, 1, self.n_nodes)
        combined_t = np.concatenate([control_t, t])
        combined_y = self.base_function(combined_t)
        control_y, y_original, _ = np.split(combined_y, [self.n_nodes, len(combined_t)])
        rand_purturb = (2 * np.random.random((self.n_nodes,)) - 1) * purturbation
        control_y += rand_purturb
        nodes = np.stack((control_t, control_y))
        curve = Curve.from_nodes(nodes)
        y_purturb = curve.evaluate_multi(t[: self.context_length])[1]
        y_original = y_original[self.context_length :]
        y = np.concatenate((y_purturb, y_original))
        # y = curve.evaluate_multi(t)[1]
    
        return y

    def __getitem__(self, idx) -> Dict:
        '''
        Returns a random trajectory  
        '''
        t = np.linspace(0, 1, self.n_frames)
        purturbation = np.ones(self.n_nodes) * self.purturbation
        purturbation[0] = 0
        purturbation[-1] = 0
        mid = (self.n_nodes) // 2
        purturbation[mid - self.n_nodes // 4 : mid] = 0
        if self.split != "training":
            spike_idx = np.random.randint(1, self.n_nodes // 2)
            purturbation[spike_idx] = self.spike_multiplier * self.purturbation
        y = self.purturb_function(t, purturbation)
        y = torch.from_numpy(y).float()[..., None]

        if self.conditional:
            conditions = y[: self.context_length, :]
            y = y[self.context_length :, :]
            output_dict = dict(xs = y, conds=conditions)
        else:   
            output_dict = dict(xs=y)

        return output_dict

class ExponentialDataset(DoubleConeDataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__(cfg, split)
        self.alpha = cfg.alpha
        self.epsilon = 1e-5
    
    def base_function(self, t: np.ndarray) -> np.ndarray | float:
        k = np.random.random() * 2 - 1
        if k >= 0:
            y = (np.exp(k * t*self.alpha)-1) / (2 * np.exp(self.alpha)) + self.epsilon
        else:
            y = (-np.exp(-k * t * self.alpha)+1)/(2 * np.exp(self.alpha)) - self.epsilon
        return y
    
class BimodalDataset(DoubleConeDataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__(cfg, split)
        self.alpha = cfg.alpha
        # self.offset = cfg.offset

    def base_function(self, t: np.ndarray) -> np.ndarray | float:
        up = np.random.random() > 0.5
        theta = np.random.random() 
        b = (1 if theta > 0.5 else -1)*0.5 + np.random.random()
        y = 2*(b-(1 if up else -1))*t + (1 if up else -1)
        k = np.random.random()+2
        y[self.context_length:] = (1 if theta > 0.5 else -1)*np.exp(k * self.alpha * 
        (t[self.context_length:] - t[self.context_length]))+ (-1 if theta > 0.5 else 1) +y[self.context_length-1]
        return y
    

class DiagonalDataset(DoubleConeDataset):

    def base_function(self, t: np.ndarray | float) -> np.ndarray | float:
        if np.random.random() < 0.5:
            y = 0.5 - t
        else:
            y = t - 0.5
        if self.split != "training":
            y[t < 0.5] = 0
        return y

class HorizontalDataset(DoubleConeDataset):
    def base_function(self, t: np.ndarray | float) -> np.ndarray | float:
        c = np.random.random() - 0.5
        y = np.full_like(t, c)
        if self.split != "training":
            override = (t - 0.4) * (np.random.random() * 2 - 1) + c
            y[t < 0.4] = override[t < 0.4]
        return y

    def __getitem__(self, idx) -> Dict:
        t = np.linspace(0, 1, self.n_frames)
        y = self.base_function(t)
        y = torch.from_numpy(y).float()[..., None]

        output_dict = dict(xs=y)

        return output_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    cfg = DictConfig(
        {
            "n_frames": 200,
            "bazier_degree": 15,
            "purturbation": 0.05,
            "spike_multiplier": 140,
            "alpha": 1,
            "conditional": True,
            "external_cond_dim": False,
            "context_length": 100
            # "offset": 0
        }
    )
    # split = "training"
    split = "validation"
    dataset = BimodalDataset(cfg, split=split)
    dataloader = DataLoader(dataset, batch_size=20)
    for d in dataloader:
        for y in d["xs"][:, :, 0]:
            t = np.linspace(0, 1, len(y))
            col = 'b' if y[0] < 0 else 'r'
            plt.plot(t, y, color=col)
        plt.show()
        break
    plt.savefig("outputs/debug.png")