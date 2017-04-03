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
    #TODO not verified
    loss_shape = loss(gan, x).get_shape()
    gan.graph.k=tf.get_variable('k', loss_shape, initializer=tf.constant_initializer(0))
    d_loss = loss(gan, x)-gan.graph.k*loss(gan, g(z_d))
    g_loss = loss(gan, g(z_g))
    gan.graph.k += gan.graph.k + k_lambda * (gamma*loss(gan, x) - loss(gan, g(z_g)))

    #TODO the paper says argmin(d_loss) and argmin(g_loss).  Is `argmin` a hyperparam?

    return [d_loss, g_loss]
