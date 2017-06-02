import tensorflow as tf
import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock


configuration = hg.configuration
class ConfigurationTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            gan = GAN(graph = graph, ops = MockOps, config = {})
            self.assertEqual(gan.graph.x, graph.x)

