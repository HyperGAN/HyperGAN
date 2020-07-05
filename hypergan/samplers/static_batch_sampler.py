from hypergan.samplers.base_sampler import BaseSampler
import numpy as np

class StaticBatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent = self.gan.latent.next().data.clone()

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        return [
            ('generator', self.gan.generator.forward(self.latent))
        ]
