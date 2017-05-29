import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

encoder = UniformEncoder({'test':True})
class UniformEncoderTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(encoder.config.test, True)

if __name__ == "__main__":
    tf.test.main()
