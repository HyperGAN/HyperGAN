import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config(
        reduce=tf.reduce_mean, 
        reverse=False,
        discriminator=None,
        k_lambda=0.01
        ):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('reverse', reverse)
    selector.set('discriminator', discriminator)

    selector.set('create', create)
    selector.set('k_lambda', k_lambda)

    return selector.random_config()

def g(gan, z):
    #reuse variables
    with(tf.variable_scope("generator", reuse=True)):
        # call generator
        generator = hc.Config(hc.lookup_functions(gan.config.generator))
        nets = generator.create(generator, gan, z)
        return nets[0]

def encode(gan, x):
    for i, discriminator in enumerate(gan.config.discriminators):
        discriminator = hc.Config(hc.lookup_functions(discriminator))
        with(tf.variable_scope("discriminator", reuse=True)):
            ds = discriminator.create(gan, discriminator, x, x, gan.graph.xs, gan.graph.gs,prefix="d_"+str(i))
            print('--', ds)
            bs = gan.config.batch_size
            net = ds
            return tf.slice(net, [0,0],[bs, -1])


    #TODO g() not defined
    #TODO encode() not defined
    return g(gan, z)

def loss(gan, z):
    return gan.graph.x - g(gan, z)

def create(config, gan):
    x = gan.graph.x
    l_x = loss(gan, x)
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]

    z_d = tf.reshape(d_real, [gan.config.batch_size, -1])
    z_g= tf.reshape(d_fake, [gan.config.batch_size, -1])#tf.random_uniform([gan.config.batch_size, 2],-1, 1,dtype=gan.config.dtype)
    #TODO This is wrong
    z_g2= encode(gan, g(gan, tf.random_uniform([gan.config.batch_size, 2],-1, 1,dtype=gan.config.dtype)))
    g_z_d = g(gan, z_d)
    g_z_g = g(gan, z_g)
    l_z_d = loss(gan, z_d)
    l_z_g = loss(gan, z_g)
    l_z_g2 = loss(gan, z_g2)


    #TODO not verified
    loss_shape = l_z_d.get_shape()
    gan.graph.k=tf.get_variable('k', [1], initializer=tf.constant_initializer(0), dtype=tf.float32)
    d_loss = l_z_d-gan.graph.k*l_z_g
    g_loss = l_z_g2

    d_loss = tf.squeeze(d_loss)
    g_loss = tf.squeeze(g_loss)
    gamma =  g_loss / d_loss
    print("++", gamma)
    gamma_l_x = gamma*l_x
    k_loss = (gamma_l_x - l_z_g2)
    gan.graph.k += gan.graph.k + config.reduce(tf.reshape(config.k_lambda * k_loss, [-1]))
    gan.graph.measure = l_z_d + tf.abs(k_loss)

    #TODO the paper says argmin(d_loss) and argmin(g_loss).  Is `argmin` a hyperparam?

    return [d_loss, g_loss]
