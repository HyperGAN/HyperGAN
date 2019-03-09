import tensorflow as tf
import hyperchamber as hc
import inspect
import os

from ..base_discriminator import BaseDiscriminator

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
        total_layers = layers + (config.extra_layers or 0)
        padding = config.padding or "SAME"

        pe_layers = self.gan.skip_connections.get_array("progressive_enhancement")

        net = self.add_noise(net)

        if config.progressive_growing:
            stacks = ops.shape(net)[0] // gan.batch_size()
            print("Adding progressive growing mask zero", len(pe_layers)//(stacks-1))
            net *= self.progressive_growing_mask(len(pe_layers)//(stacks-1)) # handle batch stacking

        if layers > 0:
            initial_depth = max(ops.shape(net)[3], config.initial_depth or 64)

            net = config.block(self, net, initial_depth, filter=config.initial_filter or 3, padding=padding)
        for i in range(layers):
            is_last_layer = (i == layers-1)
            filters = ops.shape(net)[3]
            net = activation(net)
            net = self.layer_regularizer(net)

            if config.skip_layer_filters and (i+1) in config.skip_layer_filters:
                pass
            else:
                net = self.layer_filter(net, layer=i)
                print("[hypergan] adding layer filter", net)

            depth = filters + depth_increase
            net = config.block(self, net, depth, filter=config.filter or 3, padding=padding)
            net = self.normalize(net)

            print('[discriminator] layer', net)

        for i in range(config.extra_layers or 0):
            output_features = int(int(net.get_shape()[3]))
            net = activation(net)
            net = self.layer_regularizer(net)
            filter_size_w = min(3, ops.shape(net)[1])
            filter_size_h = min(3, ops.shape(net)[2])
            if padding == "VALID":
                filters = ops.shape(net)[3]
                depth = filters + depth_increase
                net = ops.conv2d(net, filter_size_w, filter_size_h, 1, 1, depth, padding=padding)
                print("depth", depth)
            else:
                net = ops.conv2d(net, filter_size_w, filter_size_h, 1, 1, output_features//(config.extra_layers_reduction or 1), padding=padding)
            #net = self.normalize(net)
            print('[discriminator] extra layer', net)
        k=-1

        if config.relation_layer:
            net = activation(net) 
            net = self.layer_regularizer(net) 
            net = self.relation_layer(net)

        #net = tf.reshape(net, [ops.shape(net)[0], -1])


        for i in range(config.fc_layers or 0):
            # if no layers, project with linear first
            if total_layers > 0:
                net = activation(net)
                net = self.layer_regularizer(net)
            net = ops.reshape(net, [ops.shape(net)[0], -1])
            net = ops.linear(net, config.fc_layer_size or 300)

        if final_activation:
            net = self.layer_regularizer(net)
            net = final_activation(net)

        print("[discriminator] output", net)

        return net
