import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def linear_projection(net, axis=1):
    net = linear(net, 1, scope="d_lsgan_lin_proj")
    return net

def config(
        reduce=linear_projection, 
        discriminator=None,
        labels=[[0,-1,-1]] # a,b,c in the paper
    ):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('discriminator', discriminator)

    selector.set('create', create)
    selector.set('labels', labels)

    return selector.random_config()

def create(config, gan):
    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]

    net = tf.concat([d_real, d_fake], 0)
    net = config.reduce(net, axis=1)
    s = [int(x) for x in net.get_shape()]
    net = tf.reshape(net, [s[0], -1])
    d_real = tf.slice(net, [0,0], [s[0]//2,-1])
    d_fake = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

    a,b,c = config.labels
    d_loss = tf.square(d_real - b)+tf.square(d_fake - a)
    g_loss = tf.square(d_fake - c)

    d_fake_loss = -d_fake
    d_real_loss = d_real

    gan.graph.d_fake_loss=tf.reduce_mean(d_fake_loss)
    gan.graph.d_real_loss=tf.reduce_mean(d_real_loss)

    return [d_loss, g_loss]

def echo(net, axis=1):
    return net

