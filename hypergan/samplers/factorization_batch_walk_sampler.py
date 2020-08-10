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
        direction = self.gan.latent.next()
        self.origin = direction
        self.pos = self.latent1.clone().detach()
        self.hardtanh = nn.Hardtanh()
        g_params = self.gan.latent_parameters()
        if self.latent1.shape[1] // 2 == g_params[0].shape[1]:
            #recombine a split
            g_params = [torch.cat([p1, p2], 1) for p1, p2 in zip(g_params[:len(g_params)//2], g_params[len(g_params)//2:])]

        self.eigvec = torch.svd(torch.cat(g_params, 0)).V
        self.index = 0
        self.direction = self.eigvec[:, self.index].unsqueeze(0)
        self.direction = self.direction / torch.norm(self.direction)
        self.ones = torch.ones_like(self.direction, device="cuda:0")
        self.mask = torch.cat([torch.zeros([1, direction.shape[1]//2]), torch.ones([1, direction.shape[1]//2])], dim=1).cuda()
        self.mask = torch.ones_like(self.mask).cuda()
        self.steps = 30

    def setup_ui(self, frame):
        import tkinter as tk
        button = tk.Button(frame, text="Resample", command=self.reset_pos)
        button.pack()
        self.scales = []
        self.labels = []
        for i in range(self.eigvec.shape[1]):
            self.scales += [tk.Scale(frame, command=self.update_scales, from_=-200.0, to=200.0, orient=tk.HORIZONTAL, length=200)]
            self.labels += [tk.Label(frame, text="Feature " +str(i))]
        [scale.set(0) for scale in self.scales]
        for label, scale in zip(self.labels, self.scales):
            label.pack()
            scale.pack()

    def update_scales(self, value):
        self.direction = self.eigvec[:, self.index].unsqueeze(0)
        pos = self.latent1.clone().detach()
        for i, scale in enumerate(self.scales):
            value = self.scales[i].get()
            if value == 0:
                continue
            pos += self.eigvec[:, i].unsqueeze(0) * value/100.0
        self.pos = pos

    def reset_pos(self):
        self.latent1 = self.gan.latent.next()
        self.latent2 = self.gan.latent.next()
        self.update_scales(None)

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        gan = self.gan

        self.gan.latent.z = self.pos
        self.steps += 1

        g = gan.generator.forward(self.pos)
        return [
            ('generator', g)
        ]
