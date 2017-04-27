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

def autoencode(gan, config, x, rx, prefix, reuse=False):
    gconfig = gan.config.generator_autoencode
    if('align_regularizer' in config):
        gconfig['layer_regularizer'] = config['layer_regularizer']
    generator = hc.Config(hc.lookup_functions(gconfig))

    with tf.variable_scope(prefix+"autoencode", reuse=reuse):
        net = hypergan.discriminators.pyramid_discriminator.discriminator(gan, config, x, rx, [], [], prefix)
        s = [int(x) for x in net.get_shape()]
        netx  = tf.slice(net, [0,0], [s[0]//2,-1])
        netg  = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

    with tf.variable_scope("autoencoder2", reuse=reuse):
        rx = generator.create(generator, gan, netx, prefix=prefix)[-1]
    with tf.variable_scope("autoencoder2", reuse=True):
        rg = generator.create(generator, gan, netg, prefix=prefix)[-1]
    print(rx)

    return [rx,rg]


def discriminator(gan, config, x, g, xs, gs, prefix="d_"):

    xa = gan.graph.xa
    xb = gan.graph.xb

    gab = gan.graph.gab
    gba = gan.graph.gba
    #mini = []

    autoencode(gan, config, xa, gan.graph.ga, prefix=prefix+"a")
    autoencode(gan, config, xb, gan.graph.gb, prefix=prefix+"b")

    if 'include_gaab' in config:
        rxa, rgabba = autoencode(gan, config, xa, gan.graph.gabba, prefix=prefix+"a", reuse=True)
        rxb, rgbaab = autoencode(gan, config, xb, gan.graph.gbaab, prefix=prefix+"b", reuse=True)

    if 'include_gba' in config:
        rxa, rgba = autoencode(gan, config, xa, gan.graph.gba, prefix=prefix+"a", reuse=True)
        rxb, rgab = autoencode(gan, config, xb, gan.graph.gab, prefix=prefix+"b", reuse=True)

    if('include_gs' in config):
        rxa, rga = autoencode(gan, config, xa, gan.graph.ga, prefix=prefix+"a", reuse=True)
        rxb, rgb = autoencode(gan, config, xb, gan.graph.gb, prefix=prefix+"b", reuse=True)

    #gan.graph.hx = rxa
    #gan.graph.hg = gba

    gan.graph.rxa = rxa#rgabba

    print("GAAAA", gan.graph.ga)

    # TODO: concat?
    error = [
        config.distance(xa, rxa),
        config.distance(xb, rxb),
        ]
    if('include_gs' in config):
        error += [
            config.distance(gan.graph.ga, rga),
            config.distance(gan.graph.gb, rgb),
        ]

    if 'include_gba' in config:
        #config.distance(xa, rgba),
        #config.distance(xb, rgab),
        error += [
            config.distance(gan.graph.gab, rgab),
            config.distance(gan.graph.gba, rgba),
        ]


    if 'include_gaab' in config:
        error += [
            config.distance(gan.graph.gabba, rgabba),
            config.distance(gan.graph.gbaab, rgbaab),
        ]
        #config.distance(xa, gan.graph.gabba),
        #config.distance(xb, gan.graph.gbaab)
        #config.distance(xa, rgabba),
        #config.distance(xb, rgbaab),
        #config.distance(xb, rgba),
    error = tf.concat(error, axis=1)
     
    error = tf.reshape(error, [gan.config.batch_size*2, -1])
    #error = tf.concat([error]+mini, axis=1)

    return error

def create_z_encoding(gan):
    encoders = []
    with(tf.variable_scope("encoder", reuse=False)):
        for i, encoder in enumerate(gan.config.encoders):
            encoder = hc.Config(hc.lookup_functions(encoder))
            zs, z_base = encoder.create(encoder, gan)
            encoders.append(zs)

    z_encoded = tf.concat(axis=1, values=encoders)

    return z_encoded


