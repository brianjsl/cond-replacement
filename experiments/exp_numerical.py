from datasets.numerical import DoubleConeDataset, DiagonalDataset, HorizontalDataset, ExponentialDataset, BimodalDataset, BimodalExponentialDataset, StochasticBimodalExponentialDataset
from algorithms.diffusion_forcing import DiffusionForcingNumerical
from .exp_base import BaseLightningExperiment


class NumericalExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experimentclass Function1DDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.n_frames = cfg.n_frames
        self.context_length = cfg.context_length
        np.random.seed(0)

    @abstractmethod
    def base_function(self, t: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        raise NotImplementedError

    def __len__(self):
        return 100000000

    @abstractmethod
    def __getitem__(self, idx) -> Dict:
        raise NotImplementedError
    """

    compatible_algorithms = dict(
        df_numerical=DiffusionForcingNumerical,
        fs_numerical=DiffusionForcingNumerical,
    )

    compatible_datasets = dict(
        # Planning datasets
        numerical_diagonal=DiagonalDataset,
        numerical_dcone=DoubleConeDataset,
        numerical_exp=ExponentialDataset,
        numerical_horizontal=HorizontalDataset,
        numerical_bimodal=BimodalDataset,
        numerical_bimodal_exponential=BimodalExponentialDataset,
        numerical_stochastic_bimodal_exponential=StochasticBimodalExponentialDataset,
    )
