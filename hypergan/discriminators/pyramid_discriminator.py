import tensorflow as tf
import hyperchamber as hc
from hypergan.discriminators.common import *
import os

class PyramidDiscriminator:
    def __init__(self,
            prefix='d_',
            activation='lrelu',
            block=standard_block,
            depth_increase=2,
            final_activation=None,
            first_conv_size=16,
            first_strided_conv_size=64,
            layer_regularizer='layer_norm',
            layers=5,
            resize=None,
            noise=None,
            layer_filter=None,
            progressive_enhancement=True,
            orthogonal_initializer_gain=1.0,
            fc_layers=0,
            fc_layer_size=1024,
            extra_layers=4,
            extra_layers_reduction=2,
            strided=False,
            create=None,
            batch_norm_momentum=[0.001],
            batch_norm_epsilon=[0.0001]
            ):
        selector = hc.Selector()
        selector.set("activation", activation)
        selector.set("block", block)#prelu("d_")])
        selector.set("depth_increase", depth_increase)# Size increase of D's features on each layer
        selector.set("final_activation", final_activation)
        selector.set("first_conv_size", first_conv_size)
        selector.set("first_strided_conv_size", first_conv_size)
        selector.set("layers", layers) #Layers in D

        selector.set('fc_layer_size', fc_layer_size)
        selector.set('fc_layers', fc_layers)
        selector.set('extra_layers', extra_layers)
        selector.set('extra_layers_reduction', extra_layers_reduction)
        selector.set('layer_filter', layer_filter) #add information to D
        selector.set('layer_regularizer', layer_regularizer) # Size of fully connected layers
        selector.set('orthogonal_initializer_gain', orthogonal_initializer_gain)
        selector.set('noise', noise) #add noise to input
        selector.set('progressive_enhancement', progressive_enhancement)
        selector.set('resize', resize)
        selector.set('strided', strided) #TODO: true does not work

        selector.set('batch_norm_momentum', batch_norm_momentum)
        selector.set('batch_norm_epsilon', batch_norm_epsilon)
        self.prefix = prefix
        self.config = selector.random_config()

    #TODO: arguments telescope, root_config/config confusing
    def create(gan, config, x, g, xs, gs):
        dconfig = {k[2:]: v for k, v in gan.config.items() if k[2:] in inspect.getargspec(gan.ops).args}
        ops = gan.ops(*dict(dconfig))

        activation = ops.lookup(config.activation)
        final_activation = ops.lookup(config.final_activation)
        depth_increase = config.depth_increase
        depth = config.layers
        batch_norm = config.layer_regularizer

        x, g = self.resize(config, x, g)
        net = self.combine_filter(config, x, g)
        net = self.add_noise(config, net)

        for i in range(depth):
            xg = None
            is_last_layer = (i == depth-1)
            filters = ops.shape(net)[3]
            #TODO better name for `batch_norm`?
            if i != 0:
                net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)

                # APPEND xs[i] and gs[i]
                if not is_last_layer:
                    xg = self.combine_filter(config, xs[i], gs[i])
                    xg = self.add_noise(config, xg)

            net = self.progressive_enhancement(net, xg)

            depth = filters + depth_increase
            if i == 0:
                depth = config.first_conv_size

            net = config.block(ops, net, config, depth)

            print('[discriminator] layer', net)

        for i in range(config.extra_layers):
            output_features = int(int(net.get_shape()[3]))
            net = activation(net)
            net = conv2d(net, output_features//config.extra_layers_reduction, name=prefix+'_extra_layer'+str(i), k_w=3, k_h=3, d_h=1, d_w=1, regularizer=None,gain=config.orthogonal_initializer_gain)
            print('[extra discriminator] layer', net)
        k=-1

        net = tf.reshape(net, [batch_size*2, -1])

        if final_activation or config.fc_layers > 0:
            net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)

        for i in range(config.fc_layers):
            net = activation(net)
            net = ops.linear(net, config.fc_layer_size)
            if final_activation or i < config.fc_layers - 1:
                net = ops.layer_regularizer(net, config.layer_regularizer, config.batch_norm_epsilon)

        if final_activation:
            net = final_activation(net)

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
        if config['layer_filter']:
            g_filter = tf.concat(axis=3, values=[g, config['layer_filter'](gan, x)])
            x_filter = tf.concat(axis=3, values=[x, config['layer_filter'](gan, x)])
            net = tf.concat(axis=0, values=[x_filter,g_filter] )
        else:
            net = tf.concat(axis=0, values=[tf.squeeze(x),tf.squeeze(g)])
        return net

    def add_noise(self, config, net):
        if(config['noise']):
            net += tf.random_normal(net.get_shape(), mean=0, stddev=config['noise'], dtype=gan.config.dtype)
        return net

    def progressive_enhancement(self, config, net):
        if config['progressive_enhancement']:
            net = tf.concat(axis=3, values=[net, xg])
        return net
