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
from tests.mocks import MockInput

default_config = hg.Configuration.default()

class BaseGanTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput())
            self.assertNotEqual(gan.config.description, None)

    # TODO: discuss `inputs` array as a map of hashes.  Uses first element in an unordered hash.
    def test_has_input(self):
        with self.test_session():
            inputs = MockInput()
            gan = BaseGAN(inputs = inputs)
            self.assertEqual(gan.inputs.x, inputs.x)

    def test_batch_size(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput(batch_size = 1))
            self.assertEqual(gan.batch_size(), 1)

    def test_width(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput())
            self.assertEqual(gan.width(), 32)

    def test_height(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput())
            self.assertEqual(gan.height(), 32)

    def test_create(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput())
            gan.create()
            self.assertEqual(gan.created, True)
            with self.assertRaises(ValidationException):
                gan.create()

    def test_get_config_value(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput())
            self.assertEqual(type(gan.get_config_value('generator')), hc.config.Config)
            self.assertEqual(gan.get_config_value('missing'), None)

    def test_create_component(self):
        with self.test_session():
            gan = BaseGAN(inputs = MockInput())
            encoder = gan.create_component(gan.config.encoder)
            self.assertEqual(type(encoder), hg.encoders.uniform_encoder.UniformEncoder)

if __name__ == "__main__":
    tf.test.main()
