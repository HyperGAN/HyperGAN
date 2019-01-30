import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.cramer_discriminator import CramerDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
import hypergan as hg

from hypergan.gan_component import GANComponent

from unittest.mock import MagicMock
from tests.mocks import MockDiscriminator, mock_gan, MockInput
config = {
        'initial_depth': 1,
        'activation': tf.nn.tanh,
        'layers': 3,
        'depth_increase' : 3,
        'block' : hg.discriminators.common.standard_block
        }

class CramerDiscriminatorTest(tf.test.TestCase):
    def test_config(self):
        return None # disable for now
        with self.test_session():
            gan = mock_gan()
            discriminator = CramerDiscriminator(gan, config)
            self.assertEqual(discriminator.config.activation, tf.nn.tanh)

if __name__ == "__main__":
    tf.test.main()
