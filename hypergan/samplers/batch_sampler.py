from hypergan.samplers.base_sampler import BaseSampler
import numpy as np

class BatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        return {
            'generator': self.gan.generator.forward(self.gan.latent.sample())
        }

