import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

encoder = UniformEncoder({'test':True})
class UniformEncoderTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(encoder.config.test, True)

    def test_projection(self):
        config = {
                "projections": [hg.encoders.uniform_encoder.identity],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformEncoder(config)
        with self.test_session():
            gan = hg.GAN(config={"batch_size": 2, "dtype": tf.float32}, graph={})
            projections, z = subject.create(gan)
            self.assertEqual(projections.get_shape()[1], z.get_shape()[1])

    def test_projection_twice(self):
        config = {
                "projections": [hg.encoders.uniform_encoder.identity, hg.encoders.uniform_encoder.identity],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformEncoder(config)
        with self.test_session():
            gan = hg.GAN(config={"batch_size": 2, "dtype": tf.float32}, graph={})
            projections, z = subject.create(gan)
            self.assertEqual(int(projections.get_shape()[1]), 2*int(z.get_shape()[1]))
            


if __name__ == "__main__":
    tf.test.main()
