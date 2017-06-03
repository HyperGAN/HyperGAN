from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.search.default_configurations import DefaultConfigurations

from hypergan import GAN
import hypergan as hg
import tensorflow as tf
import hyperchamber as hc
import numpy as np

from unittest.mock import MagicMock

graph = hc.Config({
    'x': tf.constant(1., shape=[32,32,32])
})

default_config = hg.Configuration.default()

class MockOps:
    def __init__(self):
        self.mock = True

class MockTrainer:
    def __init__(self):
        self.mock = True


class GanTest(tf.test.TestCase):
    def test_constructor(self):
        with self.test_session():
            gan = GAN(graph = graph, config = default_config)
            self.assertEqual(gan.graph.x, graph.x)

    def test_fails_with_no_trainer(self):
        trainer = MockTrainer()
        config = {}
        gan = GAN(graph = graph, config = default_config)
        with self.assertRaises(ValidationException):
            gan.train()

    def test_validate(self):
        with self.assertRaises(ValidationException):
            gan = GAN(graph = graph, config = {})

    def test_has_input(self):
        with self.test_session():
            gan = GAN(graph = {'x': None})
            self.assertEqual(gan.inputs[0], None)

    def test_train(self):
        with self.test_session():
            gan = GAN(graph = graph, config = default_config)
            gan.train()
            self.assertEqual(gan.step, 1)

    def test_train_updates_posterior(self):
        with self.test_session():
            gan = GAN(graph = graph, config = default_config)
            prior_g = gan.generator.weights[0].eval()
            prior_d = gan.discriminators[0].weights[0].eval()
            gan.train()
            posterior_d = gan.discriminators[0].weights[0].eval()
            posterior_g = gan.generator.weights[0].eval()
            self.assertNotEqual(posterior_g, prior_g)
            self.assertNotEqual(posterior_d, prior_d)

if __name__ == "__main__":
    tf.test.main()
