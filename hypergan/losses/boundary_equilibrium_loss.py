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

def autoencode(gan, x):
    #TODO g() not defined
    #TODO encode() not defined
    return g(encode(x))

def loss(gan, g_or_x):
    return g_or_x - autoencode(gan, g_or_x)

def create(config, gan):
    gamma = config.gamma
    l_x = loss(gan, x)
    gamma_l_x = gamma*l_x
    g_z_d = g(z_d)
    g_z_g = g(z_g)
    l_g_z_g = loss(gan, g_z_g)
    k_loss = gamma_l_x - l_g_z_g
    #TODO not verified
    loss_shape = loss(gan, x).get_shape()
    gan.graph.k=tf.get_variable('k', loss_shape, initializer=tf.constant_initializer(0))
    d_loss = loss(gan, x)-gan.graph.k*loss(gan, g_z_d)
    g_loss = loss(gan, g_z_g)

    k_lambda = g_loss / d_loss #TODO too many values, needs reduce
    gan.graph.k += gan.graph.k + k_lambda * k_loss
    gan.graph.measure = l_x + tf.abs(k_loss)

    #TODO the paper says argmin(d_loss) and argmin(g_loss).  Is `argmin` a hyperparam?

    return [d_loss, g_loss]
