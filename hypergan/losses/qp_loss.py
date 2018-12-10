#https://arxiv.org/pdf/1811.07296.pdf
import tensorflow as tf
import hyperchamber as hc
from functools import reduce

from hypergan.losses.base_loss import BaseLoss

class QPLoss(BaseLoss):


    def _create(self, d_real, d_fake):
        ops = self.ops
        config = self.config
        gan = self.gan

        pq = d_real
        pp = d_fake
        zeros = tf.zeros_like(d_fake)
        ones = tf.ones_like(d_fake)

        lam = 10.0/(reduce(lambda x,y:x*y, gan.output_shape()))
        dist = gan.l1_distance()

        dl = d_real - d_fake
        d_norm = 10 * tf.reduce_mean(tf.abs(dist), axis=[1, 2, 3])
        d_loss = tf.reduce_mean(- dl + 0.5 * dl**2 / d_norm)

        g_loss = d_real - d_fake



        return [d_loss, g_loss]

