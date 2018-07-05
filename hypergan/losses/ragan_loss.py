import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

TINY=1e-12
class RaganLoss(BaseLoss):

    def required(self):
        return "reduce".split()

    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan

        pq = d_real
        pp = d_fake
        zeros = tf.zeros_like(d_fake)
        ones = tf.ones_like(d_fake)


        d_loss = -tf.log(tf.nn.sigmoid(pq-pp)+TINY)
        g_loss = -tf.log(tf.nn.sigmoid(pp-pq)+TINY)

        return [d_loss, g_loss]

