import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc


from hypergan.losses.base_loss import BaseLoss

class SoftmaxLoss(BaseLoss):

    def required(self):
        return "reduce".split()

    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops

        net = gan.discriminator.sample

        net = config.reduce(net, axis=1)

        net = tf.reshape(net, [s[0], -1])
        d_real = tf.slice(net, [0,0], [s[0]//2,-1])
        d_fake = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

        ln_zb = tf.reduce_sum(tf.exp(-d_real))+tf.reduce_sum(tf.exp(-d_fake))
        ln_zb = tf.log(ln_zb)
        d_loss = tf.reduce_mean(d_real) + ln_zb
        g_loss = tf.reduce_mean(d_fake) + tf.reduce_mean(d_real) + ln_zb
        gan.graph.d_fake_loss=tf.reduce_mean(d_fake)
        gan.graph.d_real_loss=tf.reduce_mean(d_real)

        return [d_loss, g_loss]

