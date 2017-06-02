import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

class AlignGeneratorTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            generator = ResizeConvGenerator()
            self.assertEqual(generator.config.activation, tf.nn.tanh)

    def test_create(self):
        with self.test_session():
            generator = ResizeConvGenerator()
            config = hc.Config({
                'batch_size': 32,
                'channels': 3,
                'x_dims': [32,32],
                "z_projection_depth": 128
            })
            graph = hc.Config({
                'x': tf.constant(1., shape=[32,32,32])
            })
            gan = {
                'config': config,
                'graph': graph,
                'ops': TensorflowOps,
            }
            net = tf.constant(1., shape=[32,2])
            nets = generator.create(hc.Config(gan), net)
            self.assertEqual(len(nets), 3)

if __name__ == "__main__":
    tf.test.main()
