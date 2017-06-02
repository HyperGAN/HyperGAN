import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
import hypergan as hg

from unittest.mock import MagicMock

gan = hg.GAN(config={"batch_size": 2, "dtype": tf.float32}, graph={})
config = {'initial_depth': 1, 'channels': 3, 'activation': tf.nn.tanh, 'layers': 3, 'depth_increase' : 3, 'block' : hg.discriminators.common.standard_block}

discriminator = PyramidDiscriminator(gan, config)
class PyramidDiscriminatorTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(discriminator.config.activation, tf.nn.tanh)

    def test_create(self):
        config = hc.Config({
            'batch_size': 32,
            'channels': 3,
            'x_dims': [32,32],
            "z_projection_depth": 128
        })
        graph = hc.Config({
            'x': tf.constant(1., shape=[32,32,32,3])
        })
        gan = {
            'config': config,
            'graph': graph,
            'ops': TensorflowOps,
        }

        with self.test_session():
            net = tf.constant(1., shape=[32,32,32,3])
            net = discriminator.create(hc.Config(gan), graph.x, graph.x)
            self.assertEqual(int(net.get_shape()[1]), 112)

    def test_validate(self):
        with self.assertRaises(ValidationException):
            PyramidDiscriminator(gan, {})
        
if __name__ == "__main__":
    tf.test.main()
