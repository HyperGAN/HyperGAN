import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.gan_component import GANComponent
from hypergan.multi_component import MultiComponent
from tests.mocks import mock_gan
import hypergan as hg

from unittest.mock import MagicMock

gan = mock_gan()
class MockComponent(GANComponent):
    def create(self):
        pass

component = MockComponent(gan=gan, config={'test':True})
class GanComponentTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(component.config.test, True)

    def test_validate(self):
        with self.test_session():
            self.assertEqual(component.validate(), [])

    def test_gan(self):
        with self.test_session():
            self.assertEqual(component.gan, gan)

    def test_ops(self):
        with self.test_session():
            self.assertEqual(type(component.ops), TensorflowOps)

    def test_missing_gan(self):
        with self.assertRaises(ValidationException):
            GANComponent(config={}, gan=None)

    def test_proxy_methods(self):
        component = MockComponent(gan=gan, config={'test':True})
        with self.test_session():
            self.assertEqual(component.weights(), [])
            self.assertEqual(component.biases(), [])
            self.assertEqual(component.variables(), [])

    def test_relation_layer(self):
        component = MockComponent(gan=gan, config={'test':True})
        with self.test_session():
            constant = tf.zeros([1, 2, 2, 1])
            split = component.split_by_width_height(constant)
            self.assertEqual(len(split), 4)
            permute = component.permute(split, 2)
            self.assertEqual(len(permute), 12)

            constant = tf.zeros([1, 4, 4, 1])
            split = component.split_by_width_height(constant)
            self.assertEqual(len(split), 16)
            permute = component.permute(split, 2)
            self.assertEqual(len(permute), 240)


if __name__ == "__main__":
    tf.test.main()
