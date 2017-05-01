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

def autoencode(gan, config, x, rx, prefix, id=0, reuse=False):
    gconfig = gan.config.generator_autoencode
    if('align_regularizer' in config):
        gconfig['layer_regularizer'] = config['layer_regularizer']
    generator = hc.Config(hc.lookup_functions(gconfig))
    
    s = [int(q) for q in x.get_shape()]
    info_shape = [s[0],s[1],s[2],1]
    info = tf.ones(shape=info_shape)*id
    x = tf.concat([x,info],axis=3)
    rx = tf.concat([rx,info],axis=3)

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

    autoencode(gan, config, gan.graph.xa, gan.graph.ga, prefix=prefix)

    rxa, rga = autoencode(gan, config, gan.graph.xa, gan.graph.ga, prefix=prefix, id=0, reuse=True)
    rxb, rgb = autoencode(gan, config, gan.graph.xb, gan.graph.gb, prefix=prefix, id=1, reuse=True)
    rxabba, rgabba = autoencode(gan, config, gan.graph.xabba, gan.graph.gabba, id=0, prefix=prefix, reuse=True)
    rxbaab, rgbaab = autoencode(gan, config, gan.graph.xbaab, gan.graph.gbaab, id=1, prefix=prefix, reuse=True)

    rxba, rgba = autoencode(gan, config, gan.graph.xba, gan.graph.gba, id=0, prefix=prefix, reuse=True)
    rxab, rgab = autoencode(gan, config, gan.graph.xab, gan.graph.gab, id=1, prefix=prefix, reuse=True)

    errorg = []
    errorx = []


    if('include_gs' in config):
        errorx += [
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb)
        ]
        errorg += [
            config.distance(gan.graph.ga, rga),
            config.distance(gan.graph.gb, rgb),
        ]

    if 'include_gba' in config:
        errorx += [
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb),
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb)
            ]
        errorg += [
            config.distance(gan.graph.xba, rxba),
            config.distance(gan.graph.xab, rxab),
            config.distance(gan.graph.gab, rgab),
            config.distance(gan.graph.gba, rgba),
        ]


    if 'include_gaab' in config:
        errorx += [
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb),
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb)
            ]
        errorg += [
            config.distance(gan.graph.xabba, rxabba),
            config.distance(gan.graph.xbaab, rxbaab),
            config.distance(gan.graph.gabba, rgabba),
            config.distance(gan.graph.gbaab, rgbaab),
        ]

    if 'include_cross_distance' in config:
        errorx += [
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb),
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb)
            ]
        errorg += [
            config.distance(rxa, rxabba),
            config.distance(rxb, rxbaab),
            config.distance(rga, rgabba),
            config.distance(rgb, rgbaab),
        ]

    if 'include_cross_distance2' in config:
        errorx += [
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb),
            config.distance(gan.graph.xa, rxa),
            config.distance(gan.graph.xb, rxb)
            ]
        errorg += [
            config.distance(gan.graph.xa, rxabba),
            config.distance(gan.graph.xb, rxbaab),
            config.distance(gan.graph.ga, rgabba),
            config.distance(gan.graph.gb, rgbaab),
        ]
    errorx = tf.concat(errorx, axis=1)
    errorg = tf.concat(errorg, axis=1)
    error = tf.concat([errorx, errorg], axis=0)
     
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


