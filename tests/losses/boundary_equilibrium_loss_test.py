import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.losses.boundary_equilibrium_loss import BoundaryEquilibriumLoss
from hypergan.ops import TensorflowOps

from tests.mocks import mock_gan
from unittest.mock import MagicMock
from tests.mocks import mock_gan

loss_config = {
        'test': True, 
        'reduce':'reduce_mean', 
        'use_k':True,
        'k_lambda': 0.2,
        'gamma': 0.1,
        'type': 'wgan',
        'initial_k': 0.001,
        'labels': [0,1,0]
        }
class BoundaryEquilibriumLossTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            loss = BoundaryEquilibriumLoss(mock_gan(), loss_config)
            self.assertTrue(loss.config.test)

    def test_create(self):
        with self.test_session():
            gan = mock_gan()
            gan.create()
            loss = BoundaryEquilibriumLoss(gan, loss_config)
            d_loss, g_loss = loss.create()
            d_shape = loss.ops.shape(d_loss)
            g_shape = loss.ops.shape(g_loss)
            self.assertEqual(sum(d_shape), 1)
            self.assertEqual(sum(g_shape), 1)

    def test_metrics(self):
        with self.test_session():
            graph = mock_gan()

            gan = mock_gan()
            gan.create()
            loss = BoundaryEquilibriumLoss(gan, loss_config)
            d_loss, g_loss = loss.create()
            metrics = loss.metrics
            self.assertTrue(metrics['k'] != None)




if __name__ == "__main__":
    tf.test.main()
