import tensorflow as tf
import hyperchamber as hc
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
        if layers > 0:
            initial_depth = max(ops.shape(net)[3], config.initial_depth or 64)
            if config.skip_layer_filters and 0 in config.skip_layer_filters:
                pass
            else:
                net = self.layer_filter(net)
            net = config.block(self, net, initial_depth, filter=config.initial_filter or 3)
        for i in range(layers):
            xg = None
            is_last_layer = (i == layers-1)
            filters = ops.shape(net)[3]
            net = activation(net)
            net = self.layer_regularizer(net)

            if config.skip_layer_filters and (i+1) in config.skip_layer_filters:
                pass
            else:
                net = self.layer_filter(net)
                print("[hypergan] adding layer filter", net)

            net = self.progressive_enhancement(config, net, xg)

            depth = filters + depth_increase
            net = config.block(self, net, depth)

            print('[discriminator] layer', net)

        for i in range(config.extra_layers or 0):
            output_features = int(int(net.get_shape()[3]))
            net = activation(net)
            net = self.layer_regularizer(net)
            net = ops.conv2d(net, 3, 3, 1, 1, output_features//(config.extra_layers_reduction or 1))
            print('[discriminator] extra layer', net)
        k=-1

        if config.relation_layer:
            net = activation(net)
            net = self.layer_regularizer(net)
            net = self.relation_layer(net)

        #net = tf.reshape(net, [ops.shape(net)[0], -1])

        if final_activation or (config.fc_layers or 0) > 0:
            net = self.layer_regularizer(net)

        for i in range(config.fc_layers or 0):
            net = self.layer_regularizer(net)
            net = activation(net)
            net = ops.reshape(net, [ops.shape(net)[0], -1])
            net = ops.linear(net, config.fc_layer_size or 300)
            net = self.layer_regularizer(net)

        if final_activation:
            net = final_activation(net)

        print("[discriminator] output", net)

        return net
