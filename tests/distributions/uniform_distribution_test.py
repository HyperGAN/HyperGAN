import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock
from tests.mocks import MockDiscriminator, mock_gan

gan = mock_gan()
distribution = UniformDistribution(gan, {
    'test':True,
    "z": 2,
    "min": 0,
    "max": 1
})
class UniformDistributionTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(distribution.config.test, True)

    def test_projection(self):
        config = {
                "projections": [hg.distributions.uniform_distribution.identity],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformDistribution(gan, config)
        with self.test_session():
            projections = subject.create()
            self.assertEqual(subject.ops.shape(projections)[1], 2)

    def test_projection_twice(self):
        config = {
                "projections": ['identity', 'identity'],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformDistribution(gan, config)
        with self.test_session():
            projections = subject.create()
            self.assertEqual(int(projections.get_shape()[1]), len(config['projections'])*config['z'])
            
    def test_projection_gaussian(self):
        config = {
                "projections": ['identity', 'gaussian'],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformDistribution(gan, config)
        with self.test_session():
            projections = subject.create()
            self.assertEqual(int(projections.get_shape()[1]), len(config['projections'])*config['z'])
 
if __name__ == "__main__":
    tf.test.main()
