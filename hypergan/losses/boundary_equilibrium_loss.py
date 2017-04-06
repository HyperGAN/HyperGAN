import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc
from hypergan.losses.common import *

def config(
        reduce=tf.reduce_mean, 
        reverse=False,
        discriminator=None,
        k_lambda=0.01,
        labels=[[0,-1,-1]],
        initial_k=0,
        gradient_penalty=False
        ):
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
    selector.set('use_k', [True, False])

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

    l_x = d_real#loss(gan, x)
    print("}XL", l_x)
    gan.graph.k = tf.get_variable('k', [1], initializer=tf.constant_initializer(config.initial_k), dtype=tf.float32)
    if config.use_k:
        l_g_zd = gan.graph.k*d_fake#loss(gan, g_zd)*gan.graph.k
    else:
        l_g_zd = d_fake#loss(gan, g_zd)*gan.graph.k
        
    if config.type == 'wgan':
        d_loss = l_x-l_g_zd
    else:
        if config.use_k:
            d_loss = tf.square(l_x - b)+gan.graph.k*tf.square(d_fake - a)
        else:
            d_loss = tf.square(l_x - b)+tf.square(d_fake - a)
    d_loss = tf.reduce_mean(d_loss, axis=1)

    if config.gradient_penalty:
        d_loss += gradient_penalty(gan, config.gradient_penalty)

    if config.type == 'wgan':
        g_loss = d_fake
    else:
        g_loss = tf.square(d_fake - c)
    g_loss = tf.reduce_mean(g_loss, axis=1)

    #TODO not verified
    loss_shape = g_loss.get_shape()

    gamma =  g_loss / d_loss
    if config.use_k:
        gamma_l_x = gamma*tf.reduce_mean(l_x, axis=1)
    else:
        gamma_l_x = tf.reduce_mean(l_x, axis=1)
    k_loss = tf.reduce_mean(gamma_l_x - g_loss, axis=0)
    gan.graph.update_k = tf.assign(gan.graph.k, gan.graph.k + config.k_lambda * k_loss)
    measure = tf.reduce_mean(tf.reduce_mean(l_x + tf.abs(k_loss),axis=1), axis=0)
    gan.graph.measure = measure
    gan.graph.gamma = tf.reduce_mean(gamma, axis=0)

    #TODO the paper says argmin(d_loss) and argmin(g_loss).  Is `argmin` a hyperparam?

    return [d_loss, g_loss]
