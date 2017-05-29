import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

config = hc.Config({
    'batch_size': 32,
    'channels': 3,
    'x_dims': [32,32],
    "z_projection_depth": 128
})
graph = hc.Config({
    'x': tf.constant(1., shape=[32,32,32])
})

gan = hc.Config({
    'config': config,
    'graph': graph,
    'ops': TensorflowOps,
})
generator = ResizeConvGenerator(config={
    'test': True,
    'final_depth': 4,
    'activation': tf.nn.tanh,
    'final_activation': tf.nn.tanh,
    'depth_increase': 4,
    'block': hg.generators.common.standard_block
}, gan=gan)

class ResizeConvGeneratorTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(generator.config.test, True)

    def test_create(self):
        with self.test_session():
            net = tf.constant(1., shape=[32,2])
            nets = generator.create(net)
            self.assertEqual(len(nets), 3)

    def test_initial_depth(self):
        with self.test_session():
            print(generator.depths())
            depths = generator.depths()
            self.assertEqual(depths[-1], 4)

if __name__ == "__main__":
    tf.test.main()
