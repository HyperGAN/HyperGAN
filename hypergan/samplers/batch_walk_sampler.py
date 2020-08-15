from PIL import Image
from hypergan.samplers.base_sampler import BaseSampler
import numpy as np
import torch
import torch.nn as nn
import time

class BatchWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=4, session=None):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent1 = self.gan.latent.next()
        self.latent2 = self.gan.latent.next()
        self.velocity = 15/24.0
        direction = self.gan.latent.next()
        self.origin = direction
        self.pos = self.latent1
        self.direction = direction / torch.norm(direction, p=2, dim=1, keepdim=True).expand_as(direction)
        self.hardtanh = nn.Hardtanh()
        self.ones = torch.ones_like(self.direction, device="cuda:0")
        self.mask = torch.cat([torch.zeros([1, direction.shape[1]//2]), torch.ones([1, direction.shape[1]//2])], dim=1).cuda()
        self.mask = torch.ones_like(self.mask).cuda()
        #self.mask = 1 - self.mask

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        gan = self.gan

        self.pos = self.direction * self.velocity + self.pos * self.mask + (1-self.mask) * self.origin
        self.gan.latent.z = self.pos
        mask = torch.gt(self.pos, self.ones)
        mask += torch.lt(self.pos, -self.ones)
        self.direction = self.direction + 2 * self.direction * -self.ones * mask

        g = gan.generator.forward(self.pos)
        return [
            ('generator', g)
        ]
