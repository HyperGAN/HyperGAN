from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.search.default_configurations import DefaultConfigurations

from hypergan import GAN
from hypergan.generators.resizable_generator import ResizableGenerator
import hypergan as hg
import tensorflow as tf
import hyperchamber as hc
import numpy as np

from unittest.mock import MagicMock
from tests.mocks import MockInput

default_config = hg.Configuration.default()

class MockOps:
    def __init__(self):
        self.mock = True

class MockTrainer:
    def __init__(self):
        self.mock = True


class StandardGanTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            mock_input = MockInput()
            gan = GAN(inputs = mock_input, config = default_config)
            self.assertEqual(gan.inputs.x, mock_input.x)

    def test_fails_with_no_trainer(self):
        trainer = MockTrainer()
        bad_config = hg.Configuration.default()
        bad_config['trainer'] = None
        gan = GAN(inputs = MockInput(), config = bad_config)
        with self.assertRaises(ValidationException):
            gan.step()

    def test_validate(self):
        with self.assertRaises(ValidationException):
            gan = GAN(inputs = MockInput(), config = {})

    def test_default(self):
        with self.test_session():
            gan = GAN(inputs = MockInput())
            gan.create()
            self.assertEqual(type(gan.generator), ResizableGenerator)

    def test_train(self):
        with self.test_session():
            gan = GAN(inputs = MockInput())
            gan.step()
            self.assertEqual(gan.trainer.current_step, 1)

    def test_train_updates_posterior(self):
        with self.test_session():
            gan = GAN(inputs = MockInput())
            gan.create()
            prior_g = gan.session.run(gan.generator.weights()[0])
            prior_d = gan.session.run(gan.discriminator.weights()[0])
            gan.step()
            posterior_g = gan.session.run(gan.generator.weights()[0])
            posterior_d = gan.session.run(gan.discriminator.weights()[0])
            self.assertNotEqual(posterior_g.mean(), prior_g.mean())
            self.assertNotEqual(posterior_d.mean(), prior_d.mean())


    def test_overridable_components(self):
        with self.test_session():
            gan = GAN(inputs = MockInput())
            gan.discriminator = "d_override"
            gan.generator = "g_override"
            gan.encoder = "e_override"
            gan.loss = "l_override"
            gan.trainer = "t_override"

            gan.create()

            self.assertEqual(gan.discriminator, "d_override")
            self.assertEqual(gan.generator, "g_override")
            self.assertEqual(gan.encoder, "e_override")
            self.assertEqual(gan.loss, "l_override")
            self.assertEqual(gan.trainer, "t_override")

if __name__ == "__main__":
    tf.test.main()
