import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.encoders.match_discriminator_encoder import MatchDiscriminatorEncoder
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

class MockTrainer:
    def __init__(self):
        self.mock = True

graph = hc.Config({
    'x': tf.constant(1., shape=[32,32,32])
})
gan = hg.GAN(graph = graph)

class MatchDiscriminatorEncoderTest(tf.test.TestCase):
    def test_create(self):
        with self.test_session():
            #subject = MatchDiscriminatorEncoder({})

            #z = subject.create(gan)
            #self.assertEqual(z.get_shape()[1], [])
            pass

if __name__ == "__main__":
    tf.test.main()
