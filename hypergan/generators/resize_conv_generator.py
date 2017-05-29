import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class ResizeConvGenerator(BaseGenerator):

    def required(self):
        return "final_depth activation final_activation depth_increase block".split()

    def depths(self, gan, ops):
        config = self.config
        final_depth = config.final_depth
        depths = []

        target_w = ops.shape(gan.graph.x)[0]

        w = 4 # TODO config option
        i = 0
        while w < target_w:
            w*=2
            i+=1
            depths.append(final_depth + i*config.depth_increase)
        return depths

    def create(self, gan, net):
        config = self.config
        layers = 0
        primes = [4, 4]
        nets = []
        x_dims = gan.config.x_dims
        #TODO refactor to a method on GAN object
        ops.describe("generator")

        batch_size = gan.config.batch_size
        depths = self.depths(gan, ops)
        initial_depth = depths[0]
        new_shape = [batch_size, primes[0], primes[1], initial_depth]

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)

        net = ops.linear(net, initial_depth*primes[0]*primes[1])
        net = ops.reshape(net, new_shape)

        w = ops.shape(net)[1]
        target_w = ops.shape(gan.graph.x)[0]

        while w < target_w: #TODO test
            w*=2
            layers += 1

        depth_reduction = np.float32(config.depth_reduction)

        shape = ops.shape(net)

        net = config.block(ops, net, config, shape[3])
        net = self.layer_filter(gan, config, net)

        for i in range(layers):
            s = ops.shape(net)
            is_last_layer = (i == layers-1)

            reduced_layers = shape[3]-depth_reduction
            layers = gan.config.channels if is_last_layer else reduced_layers
            resize = [min(s[1]*2, x_dims[0]), min(s[2]*2, x_dims[1])]

            net = ops.resize_images(net, resize, config.resize_image_type or 1)
            net = self.layer_filter(gan, config, net)
            net = config.block(ops, net, config, layers)

            sliced = ops.slice(net, [0,0,0,0], [-1,-1,-1, gan.config.channels])
            first3 = net if is_last_layer else sliced

            first3 = ops.layer_regularizer(first3, config.layer_regularizer, config.batch_norm_epsilon)

            first3 = final_activation(first3)

            nets.append(first3)
            size = resize[0]*resize[1]*layers
            print("[generator] layer", net, size)

        return nets

    def layer_filter(self, gan, config, net):
        if config.layer_filter:
            fltr = config.layer_filter(gan, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net
