import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class DCGANGenerator(BaseGenerator):

    def required(self):
        return []

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config
        activation = ops.lookup(config.activation or 'lrelu')

        print("[dcgan] NET IS", net)

        net = ops.linear(net, 4*4*1024)

        shape = ops.shape(net)

        net = ops.reshape(net, [shape[0],4,4,1024])

        net = activation(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, 512)
        net = self.layer_regularizer(net)
        net = activation(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, 256)
        net = self.layer_regularizer(net)
        net = activation(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, 128)
        net = self.layer_regularizer(net)
        net = activation(net)
        net = ops.deconv2d(net, 5, 5, 2, 2, gan.channels())
        net = self.layer_regularizer(net)
        net = ops.lookup('tanh')(net)

        self.sample = net
        return self.sample
