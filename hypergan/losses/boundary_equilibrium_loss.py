import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc
from hypergan.generators.resize_conv_generator import minmaxzero
from hypergan.losses.common import *

def config(
        reduce=tf.reduce_mean, 
        reverse=False,
        discriminator=None,
        k_lambda=0.01,
        labels=[[0,-1,-1]],
        initial_k=0,
        gradient_penalty=False,
        use_k=[True],
        gamma=0.75):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('reverse', reverse)
    selector.set('discriminator', discriminator)

    selector.set('create', create)
    selector.set('k_lambda', k_lambda)
    selector.set('initial_k', initial_k)
    selector.set('gradient_penalty',gradient_penalty)

    selector.set('labels', labels)
    selector.set('type', ['wgan', 'lsgan'])
    selector.set('use_k', use_k)
    selector.set('gamma', gamma)

    return selector.random_config()

def g(gan, z):
    #reuse variables
    with(tf.variable_scope("generator", reuse=True)):
        # call generator
        generator = hc.Config(hc.lookup_functions(gan.config.generator))
        nets = generator.create(generator, gan, z)
        return nets[0]

def loss(gan, x, reuse=True):
    for i, discriminator in enumerate(gan.config.discriminators):
        discriminator = hc.Config(hc.lookup_functions(discriminator))
        with(tf.variable_scope("discriminator", reuse=reuse)):
            ds = discriminator.create(gan, discriminator, x, x, gan.graph.xs, gan.graph.gs,prefix="d_"+str(i))
            bs = gan.config.batch_size
            net = ds
            net = tf.slice(net, [0,0],[bs, -1])
            print('net is', net)
            return tf.reduce_mean(net, axis=1)


def create(config, gan):
    a,b,c = config.labels
    x = gan.graph.x
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]

    d_fake = config.reduce(d_fake, axis=1)
    d_real = config.reduce(d_real, axis=1)

    gan.graph.k = tf.get_variable('k', [1], initializer=tf.constant_initializer(config.initial_k), dtype=tf.float32)
    if config.type == 'wgan':
        l_x = d_real
        l_dg =-d_fake
        l_g = d_fake
    else:
        l_x = tf.square(d_real-b)
        l_dg = tf.square(d_fake - a)
        l_g = tf.square(d_fake - c)

    if config.use_k:
        d_loss = l_x+gan.graph.k*l_dg
    else:
        d_loss = l_x+l_dg

    if config.gradient_penalty:
        d_loss += gradient_penalty(gan, config.gradient_penalty)

    gamma = config.gamma * tf.ones_like(d_fake)

    if config.use_k:
        gamma_l_x = gamma*l_x
    else:
        gamma_l_x = l_x
    k_loss = tf.reduce_mean(gamma_l_x - l_g, axis=0)
    gan.graph.update_k = tf.assign(gan.graph.k, minmaxzero(gan.graph.k + config.k_lambda * k_loss))
    measure = tf.reduce_mean(l_x + tf.abs(k_loss), axis=0)
    gan.graph.measure = measure

    gan.graph.gamma = tf.reduce_mean(gamma, axis=0)


    return [d_loss, l_g]
