import tensorflow as tf
import hyperchamber as hc
from hypergan.discriminators.common import *
import inspect
import os

from .base_discriminator import BaseDiscriminator

class PyramidDiscriminator(BaseDiscriminator):

    def required(self):
        return "activation layers block depth_increase initial_depth".split()

    def create(self):
        config = self.config
        gan = self.gan
        ops = self.ops

        x = gan.inputs.x
        g = gan.generator.sample

        x, g = self.resize(config, x, g)
        net = self.combine_filter(config, x, g)
        net = self.build(net)
        self.sample = net
        return net

    def build(self, net):
        config = self.config
        gan = self.gan
        ops = self.ops

        layers = config.layers
        depth_increase = config.depth_increase
        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)


        x = gan.inputs.x
        g = gan.generator.sample


        net = self.add_noise(config, net)

        for i in range(layers):
            xg = None
            is_last_layer = (i == layers-1)
            filters = ops.shape(net)[3]
            #TODO better name for `batch_norm`?
            if i != 0:
                net = self.layer_regularizer(net)

                # APPEND xs[i] and gs[i]
                if not is_last_layer:
                    shape = ops.shape(net)
                    small_x = ops.resize_images(x, [shape[1], shape[2]], 1)
                    small_g = ops.resize_images(g, [shape[1], shape[2]], 1)
                    xg = self.combine_filter(config, small_x, small_g)
                    xg = self.add_noise(config, xg)

            net = self.progressive_enhancement(config, net, xg)

            depth = filters + depth_increase
            if i == 0:
                depth = config.initial_depth

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
            net = self.layer_regularizer(net)
            net = activation(net)
            net = self.relation_layer(net)

        net = tf.reshape(net, [ops.shape(net)[0], -1])

        if final_activation or (config.fc_layers or 0) > 0:
            net = self.layer_regularizer(net)

        for i in range(config.fc_layers or 0):
            net = activation(net)
            net = ops.linear(net, config.fc_layer_size)
            if final_activation or i < config.fc_layers - 1:
                net = self.layer_regularizer(net)

        if final_activation:
            net = final_activation(net)

        return net

    def reuse(self, net):
        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()
        return net


    def resize(self, config, x, g):
        if(config.resize):
            # shave off layers >= resize 
            def should_ignore_layer(layer, resize):
                return int(layer.get_shape()[1]) > config['resize'][0] or \
                       int(layer.get_shape()[2]) > config['resize'][1]

            xs = [px for px in xs if not should_ignore_layer(px, config['resize'])]
            gs = [pg for pg in gs if not should_ignore_layer(pg, config['resize'])]

            x = tf.image.resize_images(x,config['resize'], 1)
            g = tf.image.resize_images(g,config['resize'], 1)

        else:
            return x, g

    def combine_filter(self, config, x, g):
        # TODO: This is standard optimization from improved GAN, cross-d feature
        if 'layer_filter' in config:
            g_filter = tf.concat(axis=3, values=[g, config['layer_filter'](gan, x)])
            x_filter = tf.concat(axis=3, values=[x, config['layer_filter'](gan, x)])
            net = tf.concat(axis=0, values=[x_filter,g_filter] )
        else:
            print("XG", x, g)
            net = tf.concat(axis=0, values=[x,g])
        return net

    def add_noise(self, config, net):
        if('noise' in config and config['noise']):
            net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise'], dtype=tf.float32)
        return net

    def progressive_enhancement(self, config, net, xg):
        if 'progressive_enhancement' in config and config.progressive_enhancement and xg is not None:
            net = tf.concat(axis=3, values=[net, xg])
        return net
