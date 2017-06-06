import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class SoftmaxLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config
        ops = self.ops

        ln_zb = tf.reduce_sum(tf.exp(-d_real), axis=1)+tf.reduce_sum(tf.exp(-d_fake), axis=1)
        ln_zb = tf.log(ln_zb)
        #TODO ln_zb = tf.reduce_log_expsum?

        d_loss = tf.reduce_mean(d_real, axis=1) + ln_zb
        g_loss = tf.reduce_mean(d_fake, axis=1) + tf.reduce_mean(d_real, axis=1) + ln_zb

        return [d_loss, g_loss]

