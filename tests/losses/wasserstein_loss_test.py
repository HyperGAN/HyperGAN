import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.losses.least_squares_loss import LeastSquaresLoss
from hypergan.losses.wasserstein_loss import WassersteinLoss
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock
from tests.mocks import mock_gan

loss_config = {'test': True, 'reduce':'reduce_mean', 'labels': [0,1,0]}
class WassersteinLossTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            loss = WassersteinLoss(hg.GAN(), loss_config)
            self.assertTrue(loss.config.test)

    def test_create(self):
        with self.test_session():
            gan = mock_gan()

            gan.create()
            loss = WassersteinLoss(gan, loss_config)
            d_loss, g_loss = loss.create()
            d_shape = gan.ops.shape(d_loss)
            g_shape = gan.ops.shape(g_loss)
            self.assertEqual(sum(d_shape), 0)
            self.assertEqual(sum(g_shape), 0)

if __name__ == "__main__":
    tf.test.main()
