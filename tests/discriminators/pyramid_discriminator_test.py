import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

discriminator = PyramidDiscriminator(prefix='test', activation=tf.nn.tanh)
class PyramidDiscriminatorTest(tf.test.TestCase):
    def testConfig(self):
        with self.test_session():
            self.assertEqual(discriminator.config.activation, tf.nn.tanh)

    def testCreate(self):
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
            self.assertEqual(int(net.get_shape()[1]), 24)
if __name__ == "__main__":
    tf.test.main()
