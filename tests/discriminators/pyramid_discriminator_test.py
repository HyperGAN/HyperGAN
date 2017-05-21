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
        with self.test_session():
            self.assertEqual(discriminator.config.activation, tf.nn.tanh)
if __name__ == "__main__":
    tf.test.main()
