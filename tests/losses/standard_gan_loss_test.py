import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.discriminators.pyramid_discriminator import PyramidDiscriminator
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock
from tests.mocks import mock_gan

from hypergan.losses.standard_loss import StandardLoss

loss_config = {'test': True, 'reduce':'reduce_mean', 'labels': [0,1,0],
        'label_smooth': 0.4, 'alpha': 0.3, 'beta': 0.2}
class StandardGanLossTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            loss = StandardLoss(hg.GAN(), loss_config)
            self.assertTrue(loss.config.test)

    def test_create(self):
        with self.test_session():
            gan = mock_gan()

            gan.create()
            loss = StandardLoss(gan, loss_config)
            d_loss, g_loss = loss.create()
            d_shape = gan.ops.shape(d_loss)
            g_shape = gan.ops.shape(g_loss)
            self.assertEqual(d_shape, [])
            self.assertEqual(g_shape, [])


if __name__ == "__main__":
    tf.test.main()
