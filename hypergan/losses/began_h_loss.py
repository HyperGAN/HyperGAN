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

def dist(x1, x2):
    bs = int(x1.get_shape()[0])
    return tf.reshape(tf.square(x1 - x2), [bs, -1])

# boundary equilibrium gan
def began(gan, config, d_real, d_fake, prefix=''):
    a,b,c = config.labels
    d_fake = config.reduce(d_fake, axis=1)
    d_real = config.reduce(d_real, axis=1)

    k = tf.get_variable(prefix+'k', [1], initializer=tf.constant_initializer(config.initial_k), dtype=config.dtype)

    if config.type == 'wgan':
        l_x = d_real
        l_dg =-d_fake
        g_loss = d_fake
    elif config.type == 'softmax': #https://arxiv.org/pdf/1704.06191.pdf
        ln_zb = tf.reduce_sum(tf.exp(-d_real))+tf.reduce_sum(tf.exp(-d_fake))
        ln_zb = tf.log(ln_zb)
        l_x = tf.reduce_mean(d_real) + ln_zb
        g_loss = tf.reduce_mean(d_fake) + tf.reduce_mean(d_real) + ln_zb
        l_dg =-tf.reduce_mean(d_fake)-tf.reduce_mean(d_real)
    else:
        l_x = tf.square(d_real-b)
        l_dg = tf.square(d_fake - a)
        g_loss = tf.square(d_fake - c)

    if config.use_k:
        d_loss = l_x+k*l_dg
    else:
        d_loss = l_x+l_dg

    if config.gradient_penalty:
        d_loss += gradient_penalty(gan, config.gradient_penalty)

    gamma = config.gamma * tf.ones_like(d_fake)

    if config.use_k:
        gamma_d_real = gamma*d_real
    else:
        gamma_d_real = d_real
    k_loss = tf.reduce_mean(gamma_d_real - d_fake, axis=0)
    update_k = tf.assign(k, k + config.k_lambda * k_loss)
    measure = tf.reduce_mean(l_x) + tf.abs(k_loss)

    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)


    if 'include_recdistance' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.rxa, gan.graph.rxabba),
            dist(gan.graph.rxb, gan.graph.rxbaab)
            ])
        g_loss += tf.reduce_mean(reconstruction)

    if 'include_recdistance2' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.xa, gan.graph.rxabba),
            dist(gan.graph.xb, gan.graph.rxbaab)
            ])
        g_loss += tf.reduce_mean(reconstruction)


    if 'include_distance' in config:
        reconstruction = tf.add_n([
            dist(gan.graph.xa, gan.graph.xabba),
            dist(gan.graph.xb, gan.graph.xbaab)
            ])
        g_loss += tf.reduce_mean(reconstruction)
        print("- - - -- - Reconstruction loss added.")

    #if 'dist' in config:
      #d_loss += lam*dist
      #g_loss += lam*dist
    #if 'softmax' in config:
    #    #d_r = gan.graph.encoder_xb
    #    #d_f = gan.graph.encoder_g
    #    d_r = gan.graph.encode_xab
    #    d_f = gan.graph.encode_gab

    #    ln_zb = tf.reduce_sum(tf.exp(-d_r))+tf.reduce_sum(tf.exp(-d_f))
    #    ln_zb = tf.log(ln_zb)

    #    d_loss += tf.reduce_mean(d_r) + ln_zb
    #    g_loss += tf.reduce_mean(d_f) + tf.reduce_mean(d_r) + ln_zb
    # 
    #    d_r = gan.graph.encode_xba
    #    d_f = gan.graph.encode_gba

    #    ln_zb = tf.reduce_sum(tf.exp(-d_r))+tf.reduce_sum(tf.exp(-d_f))
    #    ln_zb = tf.log(ln_zb)

    #    d_loss += tf.reduce_mean(d_r) + ln_zb
    #    g_loss += tf.reduce_mean(d_f) + tf.reduce_mean(d_r) + ln_zb
 
    return [k, update_k, measure, d_loss, g_loss]


def create(config, gan):
    x = gan.graph.x
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]
    k, update_k, measure, d_loss, g_loss = began(gan, config, d_real, d_fake)
    gan.graph.measure = measure
    gan.graph.k = k
    gan.graph.update_k = update_k

    gan.graph.gamma = config.gamma


    return [d_loss, g_loss]
