import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.discriminators.autoencoder_discriminator import AutoencoderDiscriminator
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
        'block' : hg.discriminators.common.standard_block,
        'distance': 'l2_distance'
        }

class AutoencoderDiscriminatorTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            gan = mock_gan()
            discriminator = AutoencoderDiscriminator(gan, config)
            self.assertEqual(discriminator.config.activation, tf.nn.tanh)

    def test_create(self):
        with self.test_session():
            remove_d_config = hg.Configuration.default()
            remove_d_config['discriminator'] = None
            remove_d_config['loss'] = None
            remove_d_config['trainer'] = None
            gan = hg.GAN(config = remove_d_config, inputs = MockInput())
            discriminator = AutoencoderDiscriminator(gan, config)
            gan.create()
            net = discriminator.create()
            self.assertEqual(int(net.get_shape()[1]), 32)

    def test_reuse(self):
        graph = hc.Config({
            'x': tf.constant(1., shape=[32,32,32,3])
        })

        with self.test_session():
            remove_d_config = hg.Configuration.default()
            remove_d_config['discriminator'] = None
            remove_d_config['loss'] = None
            remove_d_config['trainer'] = None
            gan = hg.GAN(config = remove_d_config, inputs = MockInput())
            discriminator = AutoencoderDiscriminator(gan, config)
            gan.create()
            net = discriminator.create()
            v_count  = len(discriminator.variables())
            net = discriminator.reuse(gan.inputs.x)
            v_count2  = len(discriminator.variables())
            print('net',net.get_shape())
            self.assertEqual(int(net.get_shape()[1]), 32)
            self.assertEqual(v_count, v_count2)

if __name__ == "__main__":
    tf.test.main()
