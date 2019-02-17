import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class LogisticLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        config = self.config

        d_loss = tf.nn.softplus(-d_real) + tf.nn.softplus(d_fake)
        g_loss = tf.nn.softplus(-d_fake)

        return [d_loss, g_loss]
