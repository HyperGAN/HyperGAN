import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class ResizeConvGenerator(BaseGenerator):

    def required(self):
        return "final_depth activation final_activation depth_increase block".split()

    def depths(self):
        gan = self.gan
        ops = self.ops
        config = self.config
        final_depth = config.final_depth-config.depth_increase
        depths = []

        print("DEPTHS", gan.inputs)
        target_w = gan.width()

        w = 8
        i = 0

        depths.append(final_depth)
        while w < target_w:
            w*=2
            i+=1
            depths.append(final_depth + i*config.depth_increase)
        depths[0]=self.gan.channels()
        depths.reverse()
        return depths

    def create(self):
        gan = self.gan
        ops = self.ops
        ops.describe("generator")
        return self.build(gan.encoder.sample)

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config

        primes = [4, 4]
        nets = []

        depths = self.depths()
        initial_depth = depths[0]
        new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)

        print(gan)
        net = ops.linear(net, initial_depth*primes[0]*primes[1])
        print("RESHAPE", net, new_shape)
        net = ops.reshape(net, new_shape)

        if config.relation_layer:
            net = activation(net)
            net = self.relation_layer(net)
            net = self.layer_regularizer(net)

        depth_reduction = np.float32(config.depth_reduction)

        shape = ops.shape(net)

        net = config.block(self, net, shape[3])
        net = self.layer_filter(gan, config, net)
        print("CREATING GENERATOR")

        for i, depth in enumerate(depths):
            s = ops.shape(net)
            is_last_layer = (i == len(depths)-1)

            resize = [min(s[1]*2, gan.height()), min(s[2]*2, gan.width())]

            net = ops.resize_images(net, resize, config.resize_image_type or 1)
            net = self.layer_filter(gan, config, net)
            net = config.block(self, net, depth)

            sliced = ops.slice(net, [0,0,0,0], [-1,-1,-1, gan.channels()])
            first3 = net if is_last_layer else sliced

            first3 = self.layer_regularizer(first3)

            first3 = final_activation(first3)

            nets.append(first3)
            size = resize[0]*resize[1]*depth
            print("[generator] layer", net, size)

        print("NET_ are", nets, depths)
        self.sample = nets[-1]
        return self.sample

    def reuse(self, net):
        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()
        return net

    def layer_filter(self, gan, config, net):
        if config.layer_filter:
            fltr = config.layer_filter(gan, net)
            if fltr is not None:
                net = ops.concat(axis=3, values=[net, fltr])
        return net
