import tensorflow as tf
import hyperchamber as hc
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import os
import hypergan
from hypergan.discriminators.common import *

import hypergan.discriminators.minibatch_discriminator as minibatch

def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return a-b


def config(
        activation=lrelu,
        block=standard_block,
        block_repeat_count=1,
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
    selector.set("block", block)#prelu("d_")])
    selector.set("block_repeat_count", block_repeat_count)#prelu("d_")])
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

def f(gan, config, prefix, x, g2, reuse=False):

    print("XG2", x, g2)
    with tf.variable_scope(prefix+"autoencode", reuse=reuse):
        net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, config, x, g2, [], [], prefix)
        s = [int(j) for j in net.get_shape()]
        dx  = tf.slice(net, [0,0], [s[0]//2,-1])
        dg  = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

    with tf.variable_scope(prefix+"autoencode", reuse=True):
        net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, config, gan.graph.x2, gan.graph.x2, [], [], prefix)
        s = [int(j) for j in net.get_shape()]
        dx2  = tf.slice(net, [0,0], [s[0]//2,-1])

    return config.distance(dx, dg) - config.distance(dx, dx2)

#TODO: arguments telescope, root_config/config confusing
def discriminator(gan, config, x, g, xs, gs, prefix='d_'):
    net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, config, x, g, xs, gs, prefix)

    print('g2', gan.graph.g2)
    fx = f(gan, config, prefix, x, gan.graph.g2[-1])
    fg = f(gan, config, prefix, g, gan.graph.g2[-1], reuse=True)
    return tf.concat(axis=0, values=[fx, fg])


