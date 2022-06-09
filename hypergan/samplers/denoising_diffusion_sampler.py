from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent
import hyperchamber as hc

import numpy as np
import time
import torch
from torch import nn

class DenoisingDiffusionSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.inputs = None
        self.z = self.gan.latent.next().clone()

    def compatible_with(gan):
        if hasattr(gan, 'encoder'):
            return True
        return False


    def sample_from_model(self, coefficients, generator, n_time, x_init, latent_z):
        x = x_init
        with torch.no_grad():
            for i in reversed(range(n_time)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)
                t_time = t
                x_0 = generator(x, {"t": t_time, "z": latent_z})
                x_new = self.gan.sample_posterior(x_0, x, t)
                x = x_new.detach()
        return x

    def _sample(self):
        #if self.inputs is None:
        if self.inputs is None:
            self.x_init = torch.randn_like(self.gan.inputs.next())
            self.inputs = self.gan.inputs.next().clone().detach().cuda()
        g = self.sample_from_model(self.gan.posterior_coefficients, self.gan.generator, self.gan.T, self.x_init, self.z)
        return [
        #    ('input', self.inputs),
            #('x2', encoded_x2),
            #('g2', g2),
            ('x', self.inputs),
            ('g', g),
            #('tg', tg),
            #('g2',self.gan.generator.forward(self.gan.inputs.next(1).clone().detach(), context={"y": negy_.float().view(b,1)}))
        ]

