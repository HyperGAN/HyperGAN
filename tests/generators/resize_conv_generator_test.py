import tensorflow as tf
import hyperchamber as hc
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

generator = ResizeConvGenerator(prefix='test', activation=tf.nn.tanh)
class ResizeConvGeneratorTest(tf.test.TestCase):
    def testConfig(self):
        with self.test_session():
            self.assertEqual(generator.config.activation, tf.nn.tanh)

    def testPrefix(self):
        with self.test_session():
            self.assertEqual(generator.prefix, 'test')

    def testCreate(self):
        with self.test_session():
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
            self.assertEqual(len(nets), 4)

if __name__ == "__main__":
    tf.test.main()
