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

        if len(ops.shape(net)) == 4:
            net = self.layer_filter(net)
            if config.concat_linear:
                size = ops.shape(net)[1]*ops.shape(net)[2]*config.concat_linear_filters

                net2 = tf.reshape(net, [ops.shape(net)[0], -1])
                net2 = tf.slice(net2, [0, ops.shape(net)[1]//2+1], [ops.shape(net)[0], config.concat_linear])
                net2 = ops.linear(net2, size)
                net2 = tf.reshape(net2, [ops.shape(net)[0], ops.shape(net)[1], ops.shape(net)[2], config.concat_linear_filters])

                if config.concat_linear_regularizer:
                    net2 = self.layer_regularizer(net2)
                net2 = config.activation(net2)
                net = tf.concat([net, net2], axis=3)
                net = ops.conv2d(net, 1, 1, 1, 1, ops.shape(net)[3]//(config.extra_layers_reduction or 1))
                net = self.layer_regularizer(net)
                net = activation(net)

        else:
            net = ops.reshape(net, [ops.shape(net)[0], -1])
            primes = config.initial_dimensions or [4, 4]
            depths = self.depths(primes[0])
            initial_depth = depths[0]
            new_shape = [ops.shape(net)[0], primes[0], primes[1], initial_depth]
            net = ops.linear(net, initial_depth*primes[0]*primes[1])
            net = ops.reshape(net, new_shape)

        shape = ops.shape(net)

        if config.extra_layers:
            if padding == "VALID":
                net = tf.image.resize_images(net, [ops.shape(net)[1]+2, ops.shape(net)[2]+2],1)
            net = ops.conv2d(net, 3, 3, 1, 1, ops.shape(net)[3]//(config.extra_layers_reduction or 1), padding=padding)
            net = self.normalize(net)
            for i in range(config.extra_layers or 0):
                net = self.layer_regularizer(net)
                net = activation(net)
                if padding == "VALID":
                    net = tf.image.resize_images(net, [ops.shape(net)[1]+2, ops.shape(net)[2]+2],1)
                net = ops.conv2d(net, 3, 3, 1, 1, ops.shape(net)[3]//(config.extra_layers_reduction or 1), padding=padding)
                net = self.normalize(net)

        if config.densenet_layers:
            if padding == "VALID":
                net = tf.image.resize_images(net, [ops.shape(net)[1]+2, ops.shape(net)[2]+2],1)
            net = ops.conv2d(net, 3, 3, 1, 1, ops.shape(net)[3]//(config.extra_layers_reduction or 1), padding=padding)
            net = self.normalize(net)
            for i in range(config.densenet_layers):
                net2 = self.layer_regularizer(net)
                net2 = activation(net2)
                if padding == "VALID":
                    net2 = tf.image.resize_images(net2, [ops.shape(net2)[1]+2, ops.shape(net2)[2]+2],1)
                net2 = ops.conv2d(net2, 3, 3, 1, 1, config.densenet_filters, padding=padding)
                net2 = self.normalize(net2)
                net = tf.concat(axis=3, values=[net, net2])

 
        depths = self.depths(initial_width = shape[1])
        print("[generator] Initial depth", shape)

        if config.relation_layer:
            net = self.layer_regularizer(net)
            net = activation(net)
            net = self.relation_layer(net)
            print("[generator] relational layer", net)
        else:
            pass

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
                net = ops.deconv2d(net, 5, 5, 2, 2, depth)
                net = self.normalize(net)


            size = resize[0]*resize[1]*depth
            print("[generator] layer", net, size)

        net = self.layer_regularizer(net)
        self.add_progressive_enhancement(net)

        net = activation(net)
        resize = [gan.height(), gan.width()]

        if block != 'deconv':
            net = ops.resize_images(net, resize, config.resize_image_type or 1)
            print("POST", net)
            net = self.layer_filter(net)
            net = block(self, net, config.channels or gan.channels(), filter=config.final_filter or 3, padding=padding)
        else:
            net = self.layer_filter(net)
            if resize != [e*2 for e in ops.shape(net)[1:3]]:
                print("END SIZE", net)
                #net = ops.deconv2d(net, 5, 5, 2, 2, ops.shape(net)[3]//2)
                #net = activation(net)
                print("END SIZE2", net)
                net = ops.deconv2d(net, 5, 5, 2, 2, config.channels or gan.channels())
                print("END SIZE2", net)
                net = ops.slice(net, [0,0,0,0], [ops.shape(net)[0], resize[0], resize[1], ops.shape(net)[3]])
                print("SLICE SIZE2", net)
            else:
                net = ops.deconv2d(net, 5, 5, 2, 2, config.channels or gan.channels())


        for i in range(config.post_extra_layers or 0):
            net = activation(net)
            net = ops.conv2d(net, 3, 3, 1, 1, config.channels or gan.channels())
            print("POSTEXTRA layer", i, net)
        if final_activation:
            net = self.layer_regularizer(net)
            net = final_activation(net)

        pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")
        s = ops.shape(net)
        img_dims = [s[1],s[2]]
        self.pe_layers = [tf.image.resize_images(elem, img_dims) for i, elem in enumerate(pe_layers)] + [net]
        if gan.config.progressive_growing:
            last_layer = net * self.progressive_growing_mask(len(pe_layers))
            self.debug_pe = [self.progressive_growing_mask(i) for i, elem in enumerate(pe_layers)]
        #    net = tf.add_n(nets + [last_layer])

        return net


