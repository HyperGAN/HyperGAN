import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

from hypergan.trainers.alternating_trainer import AlternatingTrainer

config = {'d_learn_rate': 1e-3, 'g_learn_rate': 1e-3, 'd_trainer': 'rmsprop', 'g_trainer': 'adam'}
gan = hg.GAN(config={"batch_size": 2, "dtype": tf.float32}, graph={})
trainer = AlternatingTrainer(gan, config)
class AlternatingTrainerTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            self.assertEqual(trainer.config.d_learn_rate, 1e-3)

    def test_validate(self):
        with self.assertRaises(ValidationException):
            AlternatingTrainer(gan, {})

    def test_clip(self):
        with self.test_session():
            trainer.create()
            self.assertEqual(gan.graph.clip, None)

if __name__ == "__main__":
    tf.test.main()
