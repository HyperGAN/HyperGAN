import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

class StandardGanLossTest(tf.test.TestCase):
    def testConfig(self):
        with self.test_session():
            loss = ImprovedLoss({})
            self.assertEqual(loss.config.z, None)

if __name__ == "__main__":
    tf.test.main()
