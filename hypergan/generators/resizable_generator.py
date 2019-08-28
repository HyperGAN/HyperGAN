import tensorflow as tf
import numpy as np
import hyperchamber as hc
from hypergan.generators.common import *
from hypergan.generators.configurable_generator import ConfigurableGenerator

from .base_generator import BaseGenerator

class ResizableGenerator(ConfigurableGenerator):

    def required(self):
        return "final_depth".split()

    def depths(self, initial_width=4):
        gan = self.gan
        ops = self.ops
        config = self.config
        final_depth = config.final_depth
        depths = []

        target_w = gan.width()

        w = initial_width
        #ontehuas
        i = 0

        depths.append(final_depth)
        while w < target_w:
            w*=2
            depths.append(final_depth * 2**i)
            i+=1
        depths = depths[1:]
        depths.reverse()
        return depths

    def build(self, net):
        gan = self.gan
        ops = self.ops
        config = self.config

        nets = []

        block = config.block or standard_block
        padding = config.padding or "SAME"
        latent = net

        net = ops.reshape(net, [ops.shape(net)[0], -1])
        primes = config.initial_dimensions or [4, 4]
        depths = self.depths(primes[0])
        initial_depth = np.minimum(depths[0], config.max_depth or 512)
        str_depth = str(primes[0])+"*"+str(primes[1])+"*"+str(initial_depth)
        new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]
        net = self.layer_linear(net, [str_depth], {"initializer": "stylegan"})
        net = ops.reshape(net, new_shape)
        print("Generator Architecture:")
        print("linear "+str_depth)

        shape = ops.shape(net)

        depths = self.depths(initial_width = shape[1])

        depth_reduction = np.float32(config.depth_reduction)
        shape = ops.shape(net)

        filter_size = config.filter or 3


        if config.adaptive_instance_norm:
            w = latent
            for i in range(config.adaptive_instance_norm_layers or 2):
                w = self.layer_linear(w, [512], {})
            w = self.layer_identity(w, ["w"], {})
            net = self.layer_adaptive_instance_norm(net, [], {})

        for i, depth in enumerate(depths[1:]):
            s = ops.shape(net)
            resize = [min(s[1]*2, gan.height()), min(s[2]*2, gan.width())]
            net = self.layer_regularizer(net)
            self.add_progressive_enhancement(net)
            dep = np.minimum(depth, config.max_depth or 512)
            print(block + " " + str(dep))
            if block == 'deconv':
                net = self.layer_deconv(net, [dep], {"initializer": "he_normal", "avg_pool": 1, "stride": 2, "filter": 3})
            elif block == 'subpixel':
                net = self.layer_subpixel(net, [dep], {"initializer": "he_normal", "avg_pool": 1, "stride": 1, "filter": 3})
            elif block == 'resize_conv':
                net = self.layer_resize_conv(net, [dep], {"initializer": "he_normal", "avg_pool": 1, "stride": 1, "filter": 3})
            else:
                net = ops.resize_images(net, resize, config.resize_image_type or 1)
                net = block(self, net, depth, filter=filter_size, padding=padding)

            if config.adaptive_instance_norm:
                net = self.layer_adaptive_instance_norm(net, [], {})

            size = resize[0]*resize[1]*depth

        net = self.layer_regularizer(net)

        resize = [gan.height(), gan.width()]

        dep = config.channels or gan.channels()
        print(block + " " + str(dep))

        if block == 'deconv':
            if resize != [e*2 for e in ops.shape(net)[1:3]]:
                net = self.layer_deconv(net, [dep], {"initializer": "he_normal", "avg_pool": 1, "stride": 2, "filter": 3, "activation": config.final_activation or "tanh"})
                net = ops.slice(net, [0,0,0,0], [ops.shape(net)[0], resize[0], resize[1], ops.shape(net)[3]])
            else:
                net = ops.deconv2d(net, 5, 5, 2, 2, dep)

        elif block == "subpixel":
            if resize != [e*2 for e in ops.shape(net)[1:3]]:
                net = self.layer_subpixel(net, [dep], {"avg_pool": 1, "stride": 1, "filter": 3, "activation": config.final_activation or "tanh"})
                net = ops.slice(net, [0,0,0,0], [ops.shape(net)[0], resize[0], resize[1], ops.shape(net)[3]])
            else:
                net = self.layer_subpixel(net, [dep], {"avg_pool": 1, "stride": 1, "filter": 3, "activation": config.final_activation or "tanh"})

        elif block == "resize_conv":
            net = self.layer_resize_conv(net, [dep], {"w": resize[0], "h": resize[1], "avg_pool": 1, "stride": 1, "filter": 3, "activation": config.final_activation or "tanh"})

        else:
            net = ops.resize_images(net, resize, config.resize_image_type or 1)
            net = block(self, net, dep, filter=config.final_filter or 3, padding=padding)


        return net


