import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config(
        reduce=tf.reduce_mean, 
        reverse=False,
        discriminator=None,
        gamma=0.001
    ):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('reverse', reverse)
    selector.set('discriminator', discriminator)
    selector.set('gamma', gamma)

    selector.set('create', create)

    return selector.random_config()

def g(gan, z):
    #reuse variables
    with(tf.variable_scope("generator", reuse=True)):
        # call generator
        generator = hc.Config(hc.lookup_functions(gan.config.generator))
        nets = generator.create(generator, gan, z)
        return nets[0]

def autoencode(gan, z):
    #TODO g() not defined
    #TODO encode() not defined
    return g(gan, z)

def loss(gan, g_or_x):
    return g_or_x - autoencode(gan, g_or_x)

def create(config, gan):
    x = gan.graph.x
    gamma = config.gamma
    l_x = loss(gan, x)
    gamma_l_x = gamma*l_x
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]

    z_d = tf.reshape(d_real, [gan.config.batch_size, -1])
    z_g = # TODO: encode random z
    z_g2 = # TODO: encode random z
    g_z_d = g(gan, z_d)
    g_z_g = g(gan, z_g)
    g_z_g2 = loss(gan, z_g_2)
    k_loss = gamma_l_x - l_g_z_g

    #TODO not verified
    loss_shape = loss(gan, z_d).get_shape()
    gan.graph.k=tf.get_variable('k', loss_shape, initializer=tf.constant_initializer(0))
    d_loss = loss(gan, z_d)-gan.graph.k*g_z_g
    g_loss = g_z_g2

    k_lambda = g_loss / d_loss #TODO too many values, needs reduce
    gan.graph.k += gan.graph.k + k_lambda * k_loss
    gan.graph.measure = l_x + tf.abs(k_loss)

    #TODO the paper says argmin(d_loss) and argmin(g_loss).  Is `argmin` a hyperparam?

    return [d_loss, g_loss]
