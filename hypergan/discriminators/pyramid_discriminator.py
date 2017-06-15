import tensorflow as tf
import hyperchamber as hc
from hypergan.discriminators.common import *
import inspect
import os

from .base_discriminator import BaseDiscriminator

class PyramidDiscriminator(BaseDiscriminator):

    def required(self):
        return "activation layers block depth_increase initial_depth".split()

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        layers = config.layers
        depth_increase = config.depth_increase
        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)

        x, g = self.split_batch(net)

        net = self.add_noise(net)

        net = config.block(self, net, config.initial_depth or 64)
        for i in range(layers):
            xg = None
            is_last_layer = (i == layers-1)
            filters = ops.shape(net)[3]
            net = activation(net)
            #TODO better name for `batch_norm`?
            net = self.layer_regularizer(net)

            # APPEND xs[i] and gs[i]
            #if not is_last_layer:
            #    shape = ops.shape(net)
            #    small_x = ops.resize_images(x, [shape[1], shape[2]], 1)
            #    small_g = ops.resize_images(g, [shape[1], shape[2]], 1)
            #    xg = self.combine_filter(config, small_x, small_g)
            #    xg = self.add_noise(xg)

            net = self.progressive_enhancement(config, net, xg)

            depth = filters + depth_increase
            print("NET IS", net, depth, layers)
            net = config.block(self, net, depth)

            print('[discriminator] layer', net)

        for i in range(config.extra_layers or 0):
            output_features = int(int(net.get_shape()[3]))
            net = activation(net)
            net = ops.conv2d(net, 3, 3, 1, 1, output_features//config.extra_layers_reduction)
            print('[extra discriminator] layer', net)
        k=-1

        if config.relation_layer:
            net = activation(net)
            net = self.layer_regularizer(net)
            net = self.relation_layer(net)

        net = tf.reshape(net, [ops.shape(net)[0], -1])

        if final_activation or (config.fc_layers or 0) > 0:
            net = self.layer_regularizer(net)

        for i in range(config.fc_layers or 0):
            net = self.layer_regularizer(net)
            net = activation(net)
            net = ops.linear(net, config.fc_layer_size or 300)
            net = self.layer_regularizer(net)

        if final_activation:
            net = final_activation(net)

        return net
>>>>>>> develop


