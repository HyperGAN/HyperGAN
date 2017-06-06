from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.search.default_configurations import DefaultConfigurations

from hypergan.gans.base_gan import BaseGAN
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
import hypergan as hg
import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.gan_component import ValidationException, GANComponent

from unittest.mock import MagicMock

default_config = hg.Configuration.default()

class MockOps:
    def __init__(self):
        self.mock = True

class MockTrainer:
    def __init__(self):
        self.mock = True

def graph():
    return hc.Config({
        'x': tf.constant(10., shape=[1,32,32,1], dtype=tf.float32)
    })

class BaseGanTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            self.assertEqual(gan.inputs[0], g.x)

    # TODO: discuss `inputs` array as a map of hashes.  Uses first element in an unordered hash.
    def test_has_input(self):
        with self.test_session():
            gan = BaseGAN(graph = {'x': None})
            self.assertEqual(gan.inputs[0], None)

    def test_batch_size(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            self.assertEqual(gan.batch_size(), 1)

    def test_width(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            self.assertEqual(gan.width(), 32)

    def test_height(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            self.assertEqual(gan.height(), 32)

    def test_create(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            gan.create()
            self.assertEqual(gan.created, True)
            with self.assertRaises(ValidationException):
                gan.create()

    def test_get_config_value(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            self.assertEqual(type(gan.get_config_value('generator')), hc.config.Config)
            self.assertEqual(gan.get_config_value('missing'), None)

    def test_create_component(self):
        with self.test_session():
            g = graph()
            gan = BaseGAN(graph = g, config = default_config)
            encoder = gan.create_component(gan.config.encoder)
            self.assertEqual(type(encoder), hg.encoders.uniform_encoder.UniformEncoder)

if __name__ == "__main__":
    tf.test.main()
