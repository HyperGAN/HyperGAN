from PIL import Image
from hypergan.samplers.base_sampler import BaseSampler
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
        self.velocity = 2/30.0
        direction = self.gan.latent.next()
        self.origin = direction
        self.pos = self.latent1
        self.hardtanh = nn.Hardtanh()
        g_params = self.gan.latent_parameters()
        if self.latent1.shape[1] // 2 == g_params[0].shape[1]:
            #recombine a split
            g_params = [torch.cat([p1, p2], 1) for p1, p2 in zip(g_params[:len(g_params)//2], g_params[len(g_params)//2:])]
            
        self.eigvec = torch.svd(torch.cat(g_params, 0)).V
        #self.eigvec = torch.svd(list(self.gan.g_parameters())[0]).V
        self.index = 0
        self.direction = self.eigvec[:, self.index].unsqueeze(0)
        self.direction = self.direction / torch.norm(self.direction)
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
        if (self.steps - 30) % 180 == 0:
            self.index+=1
            print("Index=",self.index)
            self.direction = self.eigvec[:, self.index].unsqueeze(0)
            self.direction = self.direction / torch.norm(self.direction)

        g = gan.generator.forward(self.pos)
        return [
            ('generator', g)
        ]
