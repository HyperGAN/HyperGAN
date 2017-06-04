from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.search.default_configurations import DefaultConfigurations

from hypergan import GAN
from hypergan.generators.resize_conv_generator import ResizeConvGenerator
import hypergan as hg
import tensorflow as tf
import hyperchamber as hc
import numpy as np

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
        'x': tf.constant(1., shape=[1,32,32,1], dtype=tf.float32)
    })

class GanTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            g = graph()
            gan = GAN(graph = g, config = default_config)
            self.assertEqual(gan.inputs[0], g.x)

    def test_fails_with_no_trainer(self):
        trainer = MockTrainer()
        config = {}
        gan = GAN(graph = graph(), config = {})
        with self.assertRaises(ValidationException):
            gan.train()

    def test_validate(self):
        with self.assertRaises(ValidationException):
            gan = GAN(graph = graph(), config = {})

    def test_has_input(self):
        with self.test_session():
            gan = GAN(graph = {'x': None})
            self.assertEqual(gan.inputs[0], None)

    def test_default(self):
        with self.test_session():
            gan = GAN(graph = {'x': tf.constant(1, shape=[1,1,1,1], dtype=tf.float32)})
            gan.create()
            self.assertEqual(type(gan.generator), ResizeConvGenerator)
            self.assertEqual(type(gan.discriminators[0]), PyramidDiscriminator)
            self.assertEqual(len(gan.discriminators), 1)
            self.assertEqual(len(gan.losses), 1)

    def test_train(self):
        with self.test_session():
            gan = GAN(graph = graph())
            gan.train()
            self.assertEqual(gan.trainer.step, 1)

    def test_train_updates_posterior(self):
        with self.test_session():
            gan = GAN(graph = graph())
            gan.create()
            prior_g = gan.generator.weights()[0].eval()
            prior_d = gan.discriminators[0].weights()[0].eval()
            gan.train()
            posterior_d = gan.discriminators[0].weights()[0].eval()
            posterior_g = gan.generator.weights()[0].eval()
            self.assertNotEqual(posterior_g, prior_g)
            self.assertNotEqual(posterior_d, prior_d)

if __name__ == "__main__":
    tf.test.main()
