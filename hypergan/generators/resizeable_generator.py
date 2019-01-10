import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *

from .base_generator import BaseGenerator

class ResizeableGenerator(BaseGenerator):

    def required(self):
        return "final_depth activation depth_increase".split()

    def depths(self, initial_width=4):
        gan = self.gan
        ops = self.ops
        config = self.config
        final_depth = config.final_depth-config.depth_increase
        depths = []

        target_w = gan.width()

        w = initial_width
        #ontehuas
        i = 0

        depths.append(final_depth)
        while w < target_w:
            w*=2
            i+=1
            depths.append(final_depth + i*config.depth_increase + (gan.channels() or config.channels) - 3)
        depths = depths[1:]
        depths.reverse()
        return depths

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config

        nets = []

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)
        block = config.block or standard_block
        padding = config.padding or "SAME"


        net = ops.reshape(net, [ops.shape(net)[0], -1])
        primes = config.initial_dimensions or [4, 4]
        depths = self.depths(primes[0])
        initial_depth = depths[0]
        new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]
        net = ops.linear(net, initial_depth*primes[0]*primes[1])
        print("NET", net)
        net = ops.reshape(net, new_shape)

        shape = ops.shape(net)

        depths = self.depths(initial_width = shape[1])
        print("[generator] Initial depth", shape)

        depth_reduction = np.float32(config.depth_reduction)
        shape = ops.shape(net)

        net = self.layer_filter(net)
        filter_size = config.filter or 3
        for i, depth in enumerate(depths[1:]):
            s = ops.shape(net)
            resize = [min(s[1]*2, gan.height()), min(s[2]*2, gan.width())]
            net = self.layer_regularizer(net)
            self.add_progressive_enhancement(net)
            net = activation(net)
            if block != 'deconv':
                net = ops.resize_images(net, resize, config.resize_image_type or 1)
                net = self.layer_filter(net)
                net = block(self, net, depth, filter=filter_size, padding=padding)
                net = self.normalize(net)
            else:
                net = self.layer_filter(net)
                net = ops.deconv2d(net, 5, 5, 2, 2, np.minimum(depth, config.max_depth or 512))
                print("DECONV", net)
                net = self.normalize(net)


            size = resize[0]*resize[1]*depth
            print("[generator] layer", net, size)

        net = self.layer_regularizer(net)

        net = activation(net)
        resize = [gan.height(), gan.width()]

        if block != 'deconv':
            net = ops.resize_images(net, resize, config.resize_image_type or 1)
            net = self.layer_filter(net)
            net = block(self, net, config.channels or gan.channels(), filter=config.final_filter or 3, padding=padding)
        else:
            net = self.layer_filter(net)
            if resize != [e*2 for e in ops.shape(net)[1:3]]:
                net = ops.deconv2d(net, 5, 5, 2, 2, config.channels or gan.channels())
                net = ops.slice(net, [0,0,0,0], [ops.shape(net)[0], resize[0], resize[1], ops.shape(net)[3]])
            else:
                net = ops.deconv2d(net, 5, 5, 2, 2, config.channels or gan.channels())


        if final_activation:
            net = self.layer_regularizer(net)
            net = final_activation(net)

        return net


