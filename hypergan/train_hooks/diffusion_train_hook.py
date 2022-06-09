# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738

import torch
from inspect import isfunction
import torch.nn.functional as F
import random
from torch import nn

from hypergan.train_hooks.base_train_hook import BaseTrainHook

def cosine_beta_schedule(timesteps, s = 0.008, thres = 0.999):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, thres)

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(self, *, beta_schedule, timesteps):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent = False)

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', log(posterior_variance, eps = 1e-20))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def get_times(self, batch_size, noise_level):
        device = self.betas.device
        return torch.full((batch_size,), int(self.num_timesteps * noise_level), device = device, dtype = torch.long)

    def sample_random_times(self, batch_size):
        device = self.betas.device
        return torch.randint(0, self.num_timesteps, (batch_size,), device = device, dtype = torch.long)

    def get_learned_posterior_log_variance(self, var_interp_frac_unnormalized, x_t, t):
        # if learned variance, posterior variance and posterior log variance are predicted by the network
        # by an interpolation of the max and min log beta values
        # eq 15 - https://arxiv.org/abs/2102.09672
        min_log = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = extract(torch.log(self.betas), t, x_t.shape)
        var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

        posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        return posterior_log_variance

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


class DiffusionTrainHook(BaseTrainHook):
    """
    https://arxiv.org/abs/2206.02262
    Diffusion-GAN

    """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.T = 100
        self.diffusion = GaussianDiffusion(beta_schedule="linear", timesteps=self.T)
        self.tepl = self.build_tepl()
        self.steps = 0
    
    def build_tepl(self):
        return None

    def q(self, x, t):
        noise_level = random.uniform(0,1)
        t = self.diffusion.get_times(self.gan.batch_size(), noise_level)
        return self.diffusion.q_sample(x, t)

    def forward(self, d_loss, g_loss):
        #self.steps += 1
        #if self.steps % 4 == 0:
        #    print("Rebuilding T")
        #    self.T = self.build_T()
        #    self.tepl = self.build_tepl(T)
        self.gan.add_metric("T", self.T)
        return [None, None]

    def augment_x(self, x):
        return self.q(x, self.tepl)
    def augment_g(self, g):
        self.gan.aug_g = self.q(g, self.tepl)
        return self.gan.aug_g
