import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config():
    selector = hc.Selector()
    selector.set("reduce", [tf.reduce_mean,linear_projection])#,tf.reduce_sum,tf.reduce_logsumexp,
    selector.set('reverse', [True, False])
    selector.set('discriminator', None)

    selector.set('create', create)

    return selector.random_config()

def create(config, gan):
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]

    with tf.variable_scope("d_linear", reuse=False):
        d_real = config.reduce(d_real, axis=1)
    with tf.variable_scope("d_linear", reuse=True):
        d_fake = config.reduce(d_fake, axis=1)

    if(config.reverse):
        d_loss = d_real - d_fake
        g_loss = d_fake
    else:
        d_loss = -d_real + d_fake
        g_loss = -d_fake

    d_fake_loss = -d_fake
    d_real_loss = d_real

    gan.graph.d_fake_loss=tf.reduce_mean(d_fake_loss)
    gan.graph.d_real_loss=tf.reduce_mean(d_real_loss)

    return [d_loss, g_loss]

def linear_projection(net, axis=1):
    net = tf.squeeze(linear(net, 1, scope="d_wgan_lin_proj"))
    #net = layer_norm_1(int(net.get_shape()[0]), name='d_wgan_lin_proj_bn')(net)
    #net = tf.tanh(net)
    return net


