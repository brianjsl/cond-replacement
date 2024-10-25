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
        y_original = y_original[self.context_length:]
        y = np.concatenate((y_purturb, y_original))
    
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

    
class BimodalDataset(DoubleConeDataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__(cfg, split)
        self.offset_sigma = cfg.offset_sigma
    
    @staticmethod
    def cubic_through_two_points_with_slope(t1, y1, t2, y2, t3, slope_t3):

        # Setup the system of equations to solve for a, b, c, d
        A = np.array([
            [t1**3, t1**2, t1, 1],     # Equation for f(t1) = y1
            [t2**3, t2**2, t2, 1],     # Equation for f(t2) = y2
            [3*t3**2, 2*t3, 1, 0],     # Equation for f'(t3) = slope_t3 (derivative condition)
        ])

        # Target values for the above system (y1, y2, slope at t3, arbitrary value at t3)
        b = np.array([y1, y2, slope_t3])

        # Solve for the coefficients [a, b, c, d]
        coefficients = np.linalg.pinv(A) @ b

        # Create the cubic function using the solved coefficients
        def cubic_function(t):
            a, b, c, d = coefficients
            return a*t**3 + b*t**2 + c*t + d
    
        return cubic_function
    
    def purturb_function(self, t, purturbation: Union[np.ndarray, float] = 0.05):
        if isinstance(purturbation, float):
            purturbation = np.ones(self.n_nodes) * purturbation
        else:
            assert len(purturbation) == self.n_nodes

        control_t = np.linspace(0, t[self.context_length-1], self.n_nodes)
        combined_t = np.concatenate([control_t, t])
        combined_y = self.base_function(combined_t)

        control_y, y_original, _ = np.split(combined_y, [self.n_nodes, len(combined_t)])
        rand_purturb = (2 * np.random.random((self.n_nodes,)) - 1) * purturbation
        control_y += rand_purturb

        nodes = np.stack((control_t, control_y))
        curve = Curve.from_nodes(nodes)
        y_purturb = curve.evaluate_multi(t[: self.context_length]/(t[self.context_length-1]))[1]
        y_original = y_original[self.context_length:]
        y = np.concatenate((y_purturb, y_original))
    
        return y

    def base_function(self, t: np.ndarray) -> np.ndarray | float:
        rng = np.random.default_rng()
        offset = rng.choice([-1, 1])
        mean = rng.choice([-1, 1])
        b = rng.normal(mean, self.offset_sigma)
        frac_t = self.context_length / self.n_frames

        if (mean == offset):
            alpha = (b + offset) / (frac_t ** 2)
            slope_at_mid = 2 * alpha * frac_t
            cubic_fit = self.cubic_through_two_points_with_slope(0, offset, frac_t, b,frac_t, slope_at_mid)
            y = cubic_fit(t)
            y[self.context_length:] = alpha * (frac_t) ** 2 - offset
        else:
            alpha = (b - offset) / (frac_t ** 2)
            y = alpha * t ** 2 + offset
        return y
    

class ExponentialDataset(BimodalDataset):
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
    
    
class BimodalExponentialDataset(BimodalDataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__(cfg, split)
        self.alpha = cfg.alpha

    def base_function(self, t: np.ndarray) -> np.ndarray | float:
        t_control = np.linspace(0, 1, self.n_frames)
        start = np.random.choice([-1,1])
        theta = np.random.choice([-1.5,1.5])

        frac = t_control[self.context_length]

        b = theta + np.random.normal(0, self.cfg.offset_sigma)
        k = np.random.random()+2
        cubic_fit = self.cubic_through_two_points_with_slope(0, start, frac, b, frac, theta*k*self.alpha)
        y = cubic_fit(t_control)

        y[self.context_length:] = theta*np.exp(k * self.alpha * 
        (t_control[self.context_length:] - frac))- theta +y[self.context_length-1]
        t_idx = t * self.n_frames
        t_idx = t_idx.astype(int)
        for i in range(len(t_idx)):
            if t_idx[i] >= self.n_frames:
                t_idx[i] = self.n_frames - 1
        return y[t_idx]

    # def __getitem__(self, idx):
    #     t = np.linspace(0, 1, self.n_frames)
    #     y = self.base_function(t)
    #     y = torch.from_numpy(y).float()[..., None]
    #     output_dict = dict(xs=y)
    #     return output_dict
    
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
            "spike_multiplier": 0,
            "alpha": 1.0,
            "conditional": False,
            "external_cond_dim": 0,
            "context_length": 100,
            "offset_sigma": 0.5,
        }
    )
    split = "validation"
    dataset = BimodalExponentialDataset(cfg, split=split)
    dataloader = DataLoader(dataset, batch_size=100)
    for d in dataloader:
        print(d['xs'].shape)
        for y in d["xs"][:, :, 0]:
            t = np.linspace(0, 1, len(y))
            col = 'b' if y[0] < 0 else 'r'
            plt.plot(t, y, color=col, alpha = 0.2)
        plt.show()
        break
    plt.savefig("outputs/debug.png")
