import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.losses.boundary_equilibrium_loss import BoundaryEquilibriumLoss
from hypergan.ops import TensorflowOps

from unittest.mock import MagicMock

loss_config = {'test': True, 'reduce':'reduce_mean', 'labels': [0,1,0]}
class BoundaryEquilibriumLossTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            loss = BoundaryEquilibriumLoss(hg.GAN(), loss_config)
            self.assertTrue(loss.config.test)

    def test_create(self):
        with self.test_session():
            graph = {}
            graph['d_real'] = tf.constant(0, shape=[2,2])
            graph['d_fake'] = tf.constant(0, shape=[2,2])
            print('gcraph', graph)
            loss = BoundaryEquilibriumLoss(hg.GAN(graph=graph), loss_config)
            d_loss, g_loss = loss.create()
            d_shape = loss.ops.shape(d_loss)
            g_shape = loss.ops.shape(g_loss)
            self.assertEqual(d_shape, [])
            self.assertEqual(g_shape, [])

if __name__ == "__main__":
    tf.test.main()
