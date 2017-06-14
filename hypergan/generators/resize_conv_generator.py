import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class ResizeConvGenerator(BaseGenerator):

    def required(self):
        return "final_depth activation depth_increase".split()

    def depths(self):
        gan = self.gan
        ops = self.ops
        config = self.config
        final_depth = config.final_depth-config.depth_increase
        depths = []

        target_w = gan.width()

        w = (config.initial_dimensions or [4,4])[0]
        print("DEPTHS", gan.inputs.x)
        #ontehuas
        i = 0

        depths.append(final_depth)
        while w < target_w:
            w*=2
            i+=1
            depths.append(final_depth + i*config.depth_increase)
        depths = depths[1:]
        depths.reverse()
        return depths

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config

        primes = config.initial_dimensions or [4, 4]
        nets = []

        depths = self.depths()
        initial_depth = depths[0]
        new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)
        block = config.block or standard_block

        net = ops.linear(net, initial_depth*primes[0]*primes[1])
        net = ops.reshape(net, new_shape)

        shape = ops.shape(net)

        print("[generator] Initial depth", primes, initial_depth, config.relational_layer)

        print("+++")
        if config.relation_layer:
            print("----", "RELATIONAL LAYER")
            net = self.layer_regularizer(net)
            net = activation(net)
            net = self.relation_layer(net)
        else:
            print("!!")

        depth_reduction = np.float32(config.depth_reduction)
        shape = ops.shape(net)

        net = self.layer_filter(net)
        for i, depth in enumerate(depths[1:]):
            s = ops.shape(net)
            resize = [min(s[1]*2, gan.height()), min(s[2]*2, gan.width())]
            net = ops.resize_images(net, resize, config.resize_image_type or 1)

            net = activation(net)
            net = self.layer_regularizer(net)
            net = block(self, net, depth)

            size = resize[0]*resize[1]*depth
            print("[generator] layer", net, size)

        net = self.layer_regularizer(net)
        net = activation(net)
        resize = [gan.height(), gan.width()]
        net = ops.resize_images(net, resize, config.resize_image_type or 1)
        net = block(self, net, gan.channels())
        if final_activation:
            net = self.layer_regularizer(net)
            net = final_activation(net)

        self.sample = net
        return self.sample
