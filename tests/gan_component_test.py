import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from hypergan.gan_component import GANComponent
from hypergan.ops import TensorflowOps
import hypergan as hg

from unittest.mock import MagicMock

class MockOps:
    pass
gan = hc.Config({'ops': MockOps})
component = GANComponent(gan=gan, config={'test':True})
class GANComponentTest(tf.test.TestCase):
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

if __name__ == "__main__":
    tf.test.main()
