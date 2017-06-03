import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class LeastSquaresLoss(BaseLoss):

    def required(self):
        return "reduce".split()

    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops

        if(config.discriminator == None):
            d_real = gan.graph.d_real
            d_fake = gan.graph.d_fake
        else:
            d_real = gan.graph.d_reals[config.discriminator]
            d_fake = gan.graph.d_fakes[config.discriminator]

        net = tf.concat([d_real, d_fake], 0)
        net = config.reduce(net, axis=1)
        s = [int(x) for x in net.get_shape()]
        net = tf.reshape(net, [s[0], -1])
        d_real = tf.slice(net, [0,0], [s[0]//2,-1])
        d_fake = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

        a,b,c = config.labels
        d_loss = tf.square(d_real - b) + tf.square(d_fake - a)
        g_loss = tf.square(d_fake - c)

        if config.gradient_penalty:
            d_loss += gradient_penalty(gan, config.gradient_penalty)

        d_fake_loss = -d_fake
        d_real_loss = d_real

        d_loss = ops.squash(d_fake_loss, config.reduce)

        ##TODO REFACTOR this .  This is about reporting back numbers to the trainer
        gan.graph.d_fake_loss = ops.squash(d_fake_loss, config.reduce)
        gan.graph.d_real_loss = ops.squash(d_real_loss, config.reduce)

        d_loss = ops.squash(d_loss, config.reduce)
        g_loss = ops.squash(g_loss, config.reduce)

        return [d_loss, g_loss]
