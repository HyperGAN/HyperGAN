from PIL import Image
from hypergan.samplers.base_sampler import BaseSampler
from hypergan.viewer import GlobalViewer
import numpy as np
import torch
import time

class BatchWalkSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8, session=None):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z_start = None
        self.y = None
        self.x = None
        self.step = 0
        self.steps = []
        self.step_count = 30
        self.target = None
        self.rows = 2
        self.columns = 4
        self.set_x_step = 0
        self.latent1 = self.gan.latent.sample().data.cpu()
        self.latent2 = self.gan.latent.sample().data.cpu()

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        gan = self.gan
        self.step+=1
        if self.step > self.step_count:
            self.latent1 = self.latent2
            self.latent2 = self.gan.latent.sample().data.cpu()
            self.step = 0

        latent = self.latent2 * self.step / self.step_count + (1.0 - self.step / self.step_count) * self.latent1

        g = gan.generator.forward(latent.cuda().float())
        #    gs.append(g)
        return [
            ('generator', g)
        ]
