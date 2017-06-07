import tensorflow as tf
import hyperchamber as hc
import hypergan as hg
import numpy as np
from hypergan.losses.supervised_loss import SupervisedLoss
from hypergan.ops import TensorflowOps

from tests.mocks import mock_gan
from unittest.mock import MagicMock

loss_config = {'test': True, 'reduce':'reduce_mean', 'labels': [0,1,0]}
class SupervisedLossTest(tf.test.TestCase):
    def test_config(self):
        with self.test_session():
            loss = SupervisedLoss(mock_gan(), loss_config)
            self.assertTrue(loss.config.test)

    def test_create(self):
        with self.test_session():
            gan = mock_gan()
            gan.create()
            loss = SupervisedLoss(gan, loss_config)
            d_loss, g_loss = loss.create()
            d_shape = loss.ops.shape(d_loss)
            self.assertEqual(d_shape, [1])
            self.assertEqual(g_loss, None)

    def test_metric(self):
        with self.test_session():
            gan = mock_gan()
            gan.create()
            loss = SupervisedLoss(gan, loss_config)
            d_loss, g_loss = loss.create()
            metrics = loss.metrics
            self.assertTrue(metrics['d_class_loss'] != None)


if __name__ == "__main__":
    tf.test.main()
