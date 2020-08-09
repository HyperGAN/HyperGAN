from PIL import Image
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.viewer import GlobalViewer
import numpy as np
import random
import torch
import torch.nn as nn
import time

class FactorizationBatchWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=4, session=None):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent1 = self.gan.latent.next()
        self.latent2 = self.gan.latent.next()
        self.velocity = 5.5/24.0
        direction = self.gan.latent.next()
        self.origin = direction
        self.pos = self.latent1
        self.hardtanh = nn.Hardtanh()
        #self.mask = 1 - self.mask
        g_params = list(self.gan.g_parameters())
        params = [g_params[8]]
        params += [g_params[0]]
        print([p.shape for p in params])
        cat_params = torch.cat(params,1)
        print('cat', cat_params.shape)
        self.eigvec = torch.svd(cat_params).V
        print("Eigvec")
        print(self.eigvec.shape)
        self.index = 0
        self.direction = self.eigvec[:, self.index].unsqueeze(0)
        print(self.direction.shape)
        self.ones = torch.ones_like(self.direction, device="cuda:0")
        self.mask = torch.cat([torch.zeros([1, direction.shape[1]//2]), torch.ones([1, direction.shape[1]//2])], dim=1).cuda()
        self.mask = torch.ones_like(self.mask).cuda()
        self.steps = 30

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        gan = self.gan

        self.pos = self.direction * self.velocity + self.pos# * self.mask + (1-self.mask) * self.origin
        self.gan.latent.z = self.pos
        self.steps += 1
        if self.steps % 60 == 0:
            self.direction = -self.direction
        if (self.steps - 30) % 60 == 0:
            self.index+=1
            print("Index=",self.index)
            self.direction = self.eigvec[:, self.index].unsqueeze(0)

        g = gan.generator.forward(self.pos)
        return [
            ('generator', g)
        ]
