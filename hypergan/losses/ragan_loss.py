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

        if config.type == "least_squares":
            a,b,c = config.labels
            d_loss = 0.5*tf.square(d_real - d_fake - b) + 0.5*tf.square(d_fake - d_real - a)
            g_loss = 0.5*tf.square(d_fake - d_real - c) + 0.5*tf.square(d_real - d_fake - a)

        return [d_loss, g_loss]

