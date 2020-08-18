import numpy as np
from PIL import Image

from hypergan.batch_sample import BatchSample

class BaseSampler:
    def __init__(self, gan, samples_per_row=8, session=None):
        self.gan = gan
        self.samples_per_row = samples_per_row

    def _sample(self):
        raise "raw _sample method called.  You must override this"

    def compatible_with(gan):
        return False

    def sample(self):
        gan = self.gan

        sample = self._sample()

        return BatchSample(gan.batch_size(), sample)

