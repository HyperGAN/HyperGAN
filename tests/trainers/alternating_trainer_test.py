import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.gan_component import ValidationException
from hypergan.ops import TensorflowOps
from tests.mocks import MockDiscriminator, mock_gan

from unittest.mock import MagicMock

from hypergan.trainers.alternating_trainer import AlternatingTrainer

class AlternatingTrainerTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            config = {'d_learn_rate': 1e-3, 'g_learn_rate': 1e-3, 'd_trainer': 'rmsprop', 'g_trainer': 'adam'}
            gan = hg.GAN()
            trainer = AlternatingTrainer(gan, config)
            self.assertEqual(trainer.config.d_learn_rate, 1e-3)

    def test_validate(self):
        with self.assertRaises(ValidationException):
            gan = mock_gan()
            AlternatingTrainer(gan, {})

    def test_clip(self):
        with self.test_session():

            #trainer.create()
            #self.assertEqual(gan.graph.clip, None)
            pass

    def test_output_string(self):
        with self.test_session():
            gan = mock_gan()
            gan.create()
            config = {'d_learn_rate': 1e-3, 'g_learn_rate': 1e-3, 'd_trainer': 'rmsprop', 'g_trainer': 'adam'}
            trainer = AlternatingTrainer(gan, config)
            c = tf.constant(1)
            self.assertTrue('d_loss' in trainer.output_string({'d_loss':c}))
            self.assertTrue('g_loss' in trainer.output_string({'g_loss':c}))
            self.assertEqual(len(trainer.output_variables({'a': c, 'b': c})), 2)
if __name__ == "__main__":
    tf.test.main()
