from hypergan.samplers.base_sampler import BaseSampler
import numpy as np
import torch

class StaticBatchSampler(BaseSampler):
    def __init__(self, gan, samples_per_row=8):
        BaseSampler.__init__(self, gan, samples_per_row)
        self.latent = self.gan.latent.next().data.clone()

    def compatible_with(gan):
        if hasattr(gan, 'latent'):
            return True
        return False

    def _sample(self):
        self.gan.latent.z = self.latent
        b = self.latent.shape[0]
        y_ = torch.randint(0, len(self.gan.inputs.datasets), (b, )).to(self.latent.device)
        return [
                ('generator', self.gan.generator.forward(self.latent, context={"y": y_.float().view(b,1)}))
        ]
