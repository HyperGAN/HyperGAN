import hyperchamber as hc
import numpy as np
import torch
from .base_distribution import BaseDistribution

from ..gan_component import ValidationException
from torch.distributions import uniform
from hypergan.gan_component import ValidationException, GANComponent

TINY=1e-12

class FitnessDistribution(BaseDistribution):
    def __init__(self, gan, config):
        BaseDistribution.__init__(self, gan, config)
        klass = GANComponent.lookup_function(None, self.config['source'])
        self.source = klass(gan, config)
        self.current_channels = config["z"]
        self.current_width = 1
        self.current_height = 1
        self.current_input_size = config["z"]
        self.z = self.source.z

    def create(self):
        pass

    def next(self):
        if not hasattr(self, 'best_sample'):
            self.best_sample = self.source.next()
            self.best_sample = self.next()
        if self.config.discard_prior:
            all_samples = []
            all_scores = []
        else:
            all_samples = torch.split(self.best_sample, 1, dim=0)
            prior_best_scores = self.gan.discriminator(self.gan.generator(self.best_sample))
            all_scores = torch.split(prior_best_scores, 1, dim=0)
            all_scores = [d.mean() for d in all_scores]
        for i in range(self.config.steps or 1):
            sample = self.source.next()
            d_scores = self.gan.discriminator(self.gan.generator(sample))
            d_scores = torch.split(d_scores, 1, dim=0)
            d_scores = [d.mean() for d in d_scores]
            samples = torch.split(sample, 1, dim=0)
            all_scores += d_scores
            all_samples += samples
        all_scores = [s.item() for s in all_scores]
        sorted_idx = np.argsort(all_scores)
        if self.config.reverse:
            sorted_idx = sorted_idx[::-1]
        sorted_idx = sorted_idx[:self.gan.batch_size()]
        sorted_samples = [all_samples[idx] for idx in sorted_idx]
        self.best_sample = torch.cat(sorted_samples, dim=0)
        self.z = self.best_sample
        return self.best_sample



