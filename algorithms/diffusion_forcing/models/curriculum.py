import math
from omegaconf import DictConfig
import numpy as np
from utils.print_utils import cyan
from utils.distributed_utils import rank_zero_print


class Curriculum:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup()

    def setup(self):
        self.n_tokens = self.cfg.n_tokens
        self.context_tokens = self.cfg.context_tokens
        self.base_n_tokens = self.cfg.base_n_tokens
        self.global_steps = self.cfg.global_steps
        self.rotary_rescaling = self.cfg.rotary_rescaling
        # when to move to the next stage (compute using global_steps)
        self.global_steps_threshold = np.cumsum([0] + self.global_steps[:-1])
        self.stage = -1
        self(global_step=0)

    def __call__(self, global_step: int):
        new_stage = int(max(np.where(global_step >= self.global_steps_threshold)[0]))
        if new_stage > self.stage:
            self.stage = new_stage
            self.curr_n_tokens = self.n_tokens[self.stage]
            self.curr_context_tokens = self.context_tokens[self.stage]
            rank_zero_print(
                cyan(
                    f"Curriculum: moving to stage {self.stage}, n_tokens={self.curr_n_tokens}, n_context_tokens={self.curr_context_tokens}"
                )
            )

    @property
    def rotary_emb_kwargs(self):
        return {
            "rescaling": self.rotary_rescaling,
            "base_seq_len": self.base_n_tokens,
            "init_seq_len": self.curr_n_tokens,
        }

    @staticmethod
    def static(n_tokens: int, n_context_tokens: int):
        # n_tokens = alg_cfg.n_frames // alg_cfg.frame_stack
        new_cfg = DictConfig(
            dict(
                n_tokens=[n_tokens],
                context_tokens=[n_context_tokens],
                base_n_tokens=n_tokens,
                global_steps=[math.inf],
                rotary_rescaling="NTK",  # doesn't matter
            )
        )
        return Curriculum(new_cfg)
