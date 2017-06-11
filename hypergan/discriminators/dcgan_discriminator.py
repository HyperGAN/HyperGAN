import tensorflow as tf
import hyperchamber as hc
from hypergan.discriminators.common import *
import inspect
import os

from .base_discriminator import BaseDiscriminator

class DCGANDiscriminator(BaseDiscriminator):

    def required(self):
        return []

    def create(self):
        config = self.config
        gan = self.gan
        ops = self.ops

        x = gan.inputs.x
        g = gan.generator.sample

        net = tf.concat(axis=0, values=[x,g])
        net = self.build(net)

        self.sample = net
        return net

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        net = self.add_noise(net)
        net = ops.conv2d(net, 3, 3, 2, 2, 64)
        net = ops.lookup('batch_norm')(self, net)
        net = ops.lookup('lrelu')(net)
        for layer in range(3):
            net = ops.conv2d(net, 3, 3, 2, 2, ops.shape(net)[-1]*2)
            net = ops.lookup('batch_norm')(self, net)
            net = ops.lookup('lrelu')(net)
        net = ops.reshape(net, [ops.shape(net)[0],-1])
        net = ops.linear(net, 1)

        return net

