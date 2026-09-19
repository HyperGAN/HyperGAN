import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.distributions.uniform_distribution import UniformDistribution
from hypergan.gan_component import ValidationException

from unittest.mock import MagicMock
from tests.mocks import MockDiscriminator, mock_gan

gan = mock_gan()
distribution = UniformDistribution(gan, {
    'test':True,
    "z": 2,
    "min": 0,
    "max": 1
})
class TestUniformDistribution:
    def test_config(self):
        assert distribution.config.test == True

    def test_projection(self):
        config = {
                "projections": [hg.distributions.uniform_distribution.identity],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformDistribution(gan, config)
        projections = subject.create()
        assert subject.ops.shape(projections)[1] == 2

    def test_projection_twice(self):
        config = {
                "projections": ['identity', 'identity'],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformDistribution(gan, config)
        projections = subject.create()
        assert int(projections.get_shape()[1]) == len(config['projections'])*config['z']

    def test_projection_gaussian(self):
        config = {
                "projections": ['identity', 'gaussian'],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformDistribution(gan, config)
        projections = subject.create()
        assert int(projections.get_shape()[1]) == len(config['projections'])*config['z']
