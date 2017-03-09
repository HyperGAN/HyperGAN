#Based on https://github.com/guojunq/glsgan
import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config(
        reduce=tf.reduce_mean, 
        discriminator=None
    ):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('discriminator', discriminator)

    selector.set('lamb', 1)
    selector.set('delta', delta)
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
    orignet = tf.concat([d_real, d_fake], 1)
    net = config.reduce(net, axis=1)
    s = [int(x) for x in net.get_shape()]
    net = tf.reshape(net, [s[0], -1])
    d_real = tf.slice(net, [0,0], [s[0]//2,-1])
    d_fake = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

    #d_loss = d_real + config.lamb * config.delta(gan.graph.x, gan.graph.gs[-1])
    d_loss = d_real + 1*(delta(orignet)+d_real-d_fake)
    g_loss = d_fake

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
    net = layer_norm_1(int(net.get_shape()[0]), name='d_wgan_lin_proj_bn')(net)
    net = tf.tanh(net)
    return net

def delta(net):
    prefix='d_'
    net = tf.reshape(net, [int(net.get_shape()[0]), -1])
    net = linear(net, 2048, scope=prefix+'lin1')
    net = layer_norm_1(int(net.get_shape()[0]), name=prefix+'_lin1_bn')(net)
    net = tf.nn.relu(net)
    net = linear(net, 1, scope=prefix+'lin2')
    net = layer_norm_1(int(net.get_shape()[0]), name=prefix+'_lin2_bn')(net)
    net = tf.nn.tanh(net)
    return net

