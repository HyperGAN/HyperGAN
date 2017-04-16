import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import os
import hypergan

import hypergan.discriminators.minibatch_discriminator as minibatch

def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return a-b


def config(
        activation=lrelu,
        depth_increase=2,
        final_activation=None,
        first_conv_size=16,
        first_strided_conv_size=64,
        distance=l1_distance,
        layer_regularizer=layer_norm_1,
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
        foundation='additive',
        create=None,
        minibatch=False,
        batch_norm_momentum=[0.001],
        batch_norm_epsilon=[0.0001]
        ):
    selector = hc.Selector()
    selector.set("activation", [lrelu])#prelu("d_")])
    selector.set("depth_increase", depth_increase)# Size increase of D's features on each layer
    selector.set("final_activation", final_activation)
    selector.set("first_conv_size", first_conv_size)
    selector.set("first_strided_conv_size", first_conv_size)
    selector.set('foundation', foundation)
    selector.set("layers", layers) #Layers in D
    if create is None:
        selector.set('create', discriminator)
    else:
        selector.set('create', create)

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
    selector.set('distance', distance) #TODO: true does not work
    selector.set('minibatch', minibatch)

    selector.set('batch_norm_momentum', batch_norm_momentum)
    selector.set('batch_norm_epsilon', batch_norm_epsilon)
    return selector.random_config()

#TODO: arguments telescope, root_config/config confusing
def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, config, x, g, xs, gs, prefix)
    with tf.variable_scope("autoencoder", reuse=False):
        gconfig = gan.config.generator
        if "encoder_layer_regularizer" in gconfig:
            print("overwriting layer regularizer for encoder with ", gconfig['encoder_layer_regularizer'])
            gconfig['layer_regularizer'] = gconfig['encoder_layer_regularizer']
        generator = hc.Config(hc.lookup_functions(gconfig))

        s = [int(x) for x in net.get_shape()]
        netx  = tf.slice(net, [0,0], [s[0]//2,-1])
        netg  = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

        if config.minibatch:
            mini = net
            #mini = tf.slice(net, [0, 300], [-1, 100])
            mini = minibatch.get_minibatch_features(mini, int(mini.get_shape()[0]), gan.config.dtype, prefix, 25, 100)
        else:
            mini = []

        rx = generator.create(generator, gan, netx, prefix=prefix)[-1]
    with tf.variable_scope("autoencoder", reuse=True):
        rg = generator.create(generator, gan, netg, prefix=prefix)[-1]

    gan.graph.dx = rx
    gan.graph.dg = rg

    
    error = tf.concat([config.distance(x, rx), config.distance(g,rg)], axis=0)
    error = tf.reshape(error, [gan.config.batch_size*2, -1])
    error = tf.concat([error]+mini, axis=1)

    return error


