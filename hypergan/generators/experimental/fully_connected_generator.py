import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from ..base_generator import BaseGenerator

class FullyConnectedGenerator(BaseGenerator):

    def required(self):
        return []

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config
        activation = ops.lookup(config.activation or 'lrelu')

        print("[dcgan] NET IS", net)

        net = ops.linear(net, 1024)

        shape = ops.shape(net)
        x_shape = ops.shape(self.gan.inputs.x)
        output_size = x_shape[1]*x_shape[2]*x_shape[3]
        print("Output size", output_size)

        net = activation(net)
        net = ops.linear(net, 2*1024)
        net = activation(net)
        net = ops.linear(net, output_size)
        net = ops.lookup('tanh')(net)

        net = ops.reshape(net, output_size)

        self.sample = net
        return self.sample
