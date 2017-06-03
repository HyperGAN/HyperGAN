import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

config = {
    "activation": 'lrelu',
    'final_activation': 'tanh',
    'depth_increase': 4,
    'final_depth': 4,
    'test': True,
    'block': hg.discriminators.common.standard_block
}
graph = {
    'x': tf.constant(1., shape=[1,32,32,3])
}
gan = hg.GAN(graph=graph)
generator = ResizeConvGenerator(config=config, gan=gan)

class ResizeConvGeneratorTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(generator.config.test, True)

    def test_create(self):
        with self.test_session():
            net = tf.constant(1., shape=[1,2])
            nets = generator.create(net)
            self.assertEqual(len(nets), 3)

    def test_initial_depth(self):
        with self.test_session():
            print(generator.depths())
            depths = generator.depths()
            self.assertEqual(depths[-1], 4)

if __name__ == "__main__":
    tf.test.main()
