import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config(
        reduce=tf.reduce_mean, 
        reverse=False,
        discriminator=None
    ):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('reverse', reverse)
    selector.set('discriminator', discriminator)

    selector.set('create', create)

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

linear_projection_iterator=0
def linear_projection(net, axis=1):
    global linear_projection_iterator
    net = linear(net, 1, scope="d_wgan_lin_proj"+str(linear_projection_iterator))
    linear_projection_iterator+=1
    #net = layer_norm_1(int(net.get_shape()[0]), name='d_wgan_lin_proj_bn')(net)
    #net = tf.tanh(net)
    return net

def echo(net, axis=1):
    return net

