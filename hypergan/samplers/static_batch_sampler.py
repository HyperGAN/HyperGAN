from hypergan.samplers.base_sampler import BaseSampler
from hypergan.train_hooks.experimental.imle_train_hook import IMLETrainHook
import numpy as np
import tensorflow as tf

class StaticBatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.z = None
        self.y = None
        self.x = None
        self.rows = 4
        self.columns = samples_per_row
        self.latent = self.gan.latent.sample()

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        return {
            'generator': self.gan.generator.forward(self.latent)
        }

