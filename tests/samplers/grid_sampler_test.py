import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

from hypergan.samplers.grid_sampler import GridSampler
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from tests.mocks import mock_gan

from hypergan.distributions.uniform_distribution import UniformDistribution

class GridSamplerTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            gan = mock_gan(batch_size=32)
            gan.latent = UniformDistribution(gan, {'z':2, 'min': -1, 'max': 1, 'projections':['identity']})
            gan.latent.create()
            gan.create()

            sampler = GridSampler(gan)
            self.assertEqual(sampler._sample()['generator'].shape[-1], 1)

if __name__ == "__main__":
    tf.test.main()
