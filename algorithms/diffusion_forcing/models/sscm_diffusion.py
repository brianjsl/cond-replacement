from .diffusion import Diffusion
import numpy as np
import torch
from torch.distributions import Normal 

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, int):
        section_counts = [section_counts]
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(Diffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = Diffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class SMCDiffusion(SpacedDiffusion):
    """A diffusion model that supports FK_SMC sampling"""
    def __init__(self, particle_base_shape, **kwargs):
        super().__init__(**kwargs)

        self.T = self.num_timesteps 
        self.particle_base_shape = particle_base_shape # (C, H, W) for image
        self.particle_base_dims = [-(i+1) for i in range(len(particle_base_shape))] 

        self.clear_cache()

    def clear_cache(self):
        self.cache = {} 

    def ref_sample(self, P, device):
        return torch.randn(P, *self.particle_base_shape).to(device)
    
    def ref_log_density(self, samples):
        return Normal(loc=0, scale=1.0).log_prob(samples)
    
    def M(self, t, xtp1, extra_vals, P, model, device, **kwargs):
        raise NotImplementedError
    
    def G(self, t, xtp1, xt, extra_vals, model, debug=False, debug_info=None, **kwargs):
        raise NotImplementedError 

    def p_trans_model(self, xtp1, t, model, clip_denoised, model_kwargs):
        # compute mean and variance of p(x_t|x_{t+1}) under the model

        out = self.p_mean_variance(model, x=xtp1, t=t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        
        return {
            "mean_untwisted": out['mean'], 
            "var_untwisted": out['variance'], 
            "pred_xstart": out['pred_xstart']
        }
    
    def set_measurement_cond_fn(self, measurement_cond_fn):
        self.measurement_cond_fn = measurement_cond_fn 

    def set_measurement(self, measurement):
        self.measurement = measurement 

    def _sample_forward(self, x0, batch_size=None):
        # generate sample trajectory [x0, x1, .., xt]
        if batch_size is not None:
            new_shape = (batch_size,) + x0.shape  # create new shape tuple
            x0 = x0.expand(*new_shape)

        xts_forward = torch.empty(self.T+1, *x0.shape) # (T+1, *img_shape) or (T+1, bsz, *img_shape)
        xts_forward[0] = x0.cpu() 

        xt = x0 
        for t in range(self.T):
            xt = self.q_sample_tp1_given_t(xt, t) 
            xts_forward[t+1] = xt.cpu() 

        return xts_forward   


def _gaussian_sample(mean, variance=None, scale=None, sample_shape=th.Size([]), return_logprob=False):
    if scale is None:
        assert variance is not None 
        scale = torch.sqrt(variance)

    normal_dist = Normal(loc=mean, scale=scale)
    samples = normal_dist.sample(sample_shape)
    if return_logprob:
        log_prob = normal_dist.log_prob(samples)
        return samples, log_prob 
    return samples 


def get_to_cpu(d, k):
    if d is None:
        return None 
    v = d.get(k)
    if v is not None:
        v = v.cpu()
    return v 


def append_to_trace(item, trace):
    if item is not None:
        trace.append(item)