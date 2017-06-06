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

        net = gan.discriminators[0].sample

        net = config.reduce(net, axis=1)

        shape = ops.shape(net)
        net = tf.reshape(net, [shape[0], -1])

        #TODO can we generalize this based on `gan.inputs`?
        # split in half
        d_real = tf.slice(net, [0,0], [shape[0]//2,-1])
        d_fake = tf.slice(net, [shape[0]//2,0], [shape[0]//2,-1])

        a,b,c = config.labels
        square = ops.lookup('square')
        d_loss = square(d_real - b) + square(d_fake - a)
        g_loss = square(d_fake - c)

        if config.gradient_penalty:
            d_loss += self.gradient_penalty(gan, config.gradient_penalty)

        d_fake_loss = -d_fake
        d_real_loss = d_real

        ##TODO REFACTOR this .  This is about reporting back numbers to the trainer
        gan.graph.d_fake_loss = ops.squash(d_fake_loss, config.reduce)
        gan.graph.d_real_loss = ops.squash(d_real_loss, config.reduce)

        d_loss = ops.squash(d_fake_loss, config.reduce)
        d_loss = ops.squash(d_loss, config.reduce)
        g_loss = ops.squash(g_loss, config.reduce)

        self.sample = [d_loss, g_loss]

        return self.sample
