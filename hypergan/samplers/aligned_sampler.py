from hypergan.samplers.base_sampler import BaseSampler
from hypergan.gan_component import ValidationException, GANComponent

import numpy as np
import time
import torch

class AlignedSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.inputs = None
        self.z = self.gan.latent.next()

    def compatible_with(gan):
        if hasattr(gan, 'encoder'):
            return True
        return False

    def _sample(self):
        if self.inputs is None:
            self.inputs = self.gan.inputs.next(0).clone().detach()
        self.gan.latent.z = self.z
        b = self.z.shape[0]
        y_ = torch.randint(0, len(self.gan.inputs.datasets), (b, )).to(self.z.device)
        posy_ = torch.ones_like(y_)
        negy_ = torch.zeros_like(y_)
        return [
            ('input', self.inputs),
            ('g1',self.gan.generator.forward(self.inputs, context={"y": posy_.float().view(b,1)}))
            #('g2',self.gan.generator.forward(self.gan.inputs.next(1).clone().detach(), context={"y": negy_.float().view(b,1)}))
        ]

