import hyperchamber as hc
import numpy as np
from hypergan.gan_component import ValidationException
from hypergan.gan_component import GANComponent
from mocks import mock_gan
import hypergan as hg

from unittest.mock import MagicMock

gan = mock_gan()
class MockComponent(GANComponent):
    def create(self):
        pass

component = MockComponent(gan=gan, config={'test':True})
class TestGanComponent:
    def test_config(self):
        assert component.config.test == True

    def test_gan(self):
        assert component.gan == gan
