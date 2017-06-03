import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.gan_component import GANComponent
from hypergan.ops import TensorflowOps
import hypergan as hg

from unittest.mock import MagicMock

class MockOps:
    pass

gan = hc.Config({'ops_backend': MockOps})
component = GANComponent(gan=gan, config={'test':True})
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
            self.assertEqual(type(component.ops), MockOps)

    def test_missing_gan(self):
        with self.assertRaises(ValidationException):
            GANComponent(config={}, gan=None)

    def test_proxy_methods(self):
        component = GANComponent(gan=gan, config={'test':True})
        with self.test_session():
            self.assertEqual(component.weights, [])
            self.assertEqual(component.biases, [])
            self.assertEqual(component.variables(), [])

if __name__ == "__main__":
    tf.test.main()
