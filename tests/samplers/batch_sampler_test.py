import tensorflow as tf
import hypergan as hg
import hyperchamber as hc
import numpy as np

from hypergan.samplers.batch_sampler import BatchSampler
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from tests.mocks import mock_gan

class BatchSamplerTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            gan = mock_gan()
            gan.create()

            sampler = BatchSampler(gan)
            self.assertEqual(sampler._sample()['generator'].shape[-1], 48)

if __name__ == "__main__":
    tf.test.main()
