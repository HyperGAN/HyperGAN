import tensorflow as tf
import hyperchamber as hc
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

trainer = ProportionalControlTrainer()
class ProportionalControlTrainerTest(tf.test.TestCase):
    def testConfig(self):
        with self.test_session():
            self.assertEqual(trainer.config.d_learn_rate, 1e-3)

if __name__ == "__main__":
    tf.test.main()
