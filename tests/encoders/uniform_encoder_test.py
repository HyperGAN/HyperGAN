import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.encoders.uniform_encoder import UniformEncoder
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

gan = hg.GAN(graph = {'x': tf.constant([1,1])})
encoder = UniformEncoder(gan, {
    'test':True,
    "z": 2,
    "min": 0,
    "max": 1
})
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
        subject = UniformEncoder(gan, config)
        with self.test_session():
            projections, z = subject.create()
            self.assertEqual(projections.get_shape()[1], z.get_shape()[1])

    def test_projection_twice(self):
        config = {
                "projections": [hg.encoders.uniform_encoder.identity, hg.encoders.uniform_encoder.identity],
                "z": 2,
                "min": 0,
                "max": 1
                }
        subject = UniformEncoder(gan, config)
        with self.test_session():
            projections, z = subject.create()
            self.assertEqual(int(projections.get_shape()[1]), 2*int(z.get_shape()[1]))
            
    def test_validate(self):
        with self.assertRaises(ValidationException):
            UniformEncoder(gan, {})
 

if __name__ == "__main__":
    tf.test.main()
