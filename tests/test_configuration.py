import hyperchamber as hc
import numpy as np
import hypergan as hg
from hypergan.gan_component import ValidationException

from unittest.mock import MagicMock

class TestConfiguration:
    def test_constructor(self):
        default = hg.Configuration.default()
        assert default.trainer != None
        assert default.discriminator != None
        assert default.loss != None
