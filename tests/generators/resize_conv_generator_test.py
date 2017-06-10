import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock
from tests.mocks import MockDiscriminator, mock_gan

config = {
    "activation": 'lrelu',
    'final_activation': 'tanh',
    'depth_increase': 4,
    'final_depth': 4,
    'test': True,
    'block': hg.discriminators.common.standard_block
}
gan = mock_gan()
generator = ResizeConvGenerator(config=config, gan=gan)

class ResizeConvGeneratorTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(generator.config.test, True)

    def test_create(self):
        with self.test_session():
            gan.config['generator'] = None
            gan.config['discriminator'] = None
            gan.config['loss'] = None
            gan.config['trainer'] = None
            gan.create()
            nets = generator.create()
            self.assertEqual(generator.ops.shape(nets), [1,2,2,1])

    def test_initial_depth(self):
        with self.test_session():
            print(generator.depths())
            depths = generator.depths()
            self.assertEqual(depths[-1], 1)

    def test_layer_norm(self):
        with self.test_session():
            generator = ResizeConvGenerator(config=config, gan=gan)
            generator.ops.lookup('layer_norm')(generator.ops, tf.constant(1, shape=[1,1,1,1], dtype=tf.float32))
            self.assertNotEqual(len(generator.variables()), 0)


if __name__ == "__main__":
    tf.test.main()
