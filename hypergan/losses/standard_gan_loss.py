import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

def config():
    selector = hc.Selector()
    selector.set("reduce", [linear_projection])#tf.reduce_mean, reduce_sum, reduce_logexp work
    selector.set("label_smooth", list(np.linspace(0.15, 0.35, num=100)))
    selector.set('discriminator', None)

    selector.set('create', create)

    return selector.random_config()

def create(config, gan):
    d_real = gan.graph.d_real
    d_fake = gan.graph.d_fake

    if(config.discriminator == None):
        d_real = gan.graph.d_real
        d_fake = gan.graph.d_fake
    else:
        d_real = gan.graph.d_reals[config.discriminator]
        d_fake = gan.graph.d_fakes[config.discriminator]

    net = tf.concat([d_real, d_fake], 0)
    net = config.reduce(net, axis=1)
    s = [int(x) for x in net.get_shape()]
    d_real = tf.slice(net, [0,0], [s[0]//2,-1])
    d_fake = tf.slice(net, [s[0]//2,0], [s[0]//2,-1])

    zeros = tf.zeros_like(d_fake, dtype=gan.config.dtype)
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=zeros)
    d_loss = sigmoid_kl_with_logits(d_real, 1.-config.label_smooth)
    g_loss = tf.squeeze(g_loss)
    d_loss = tf.squeeze(d_loss)

    gan.graph.d_fake_loss=tf.reduce_mean(d_fake)
    gan.graph.d_real_loss=tf.reduce_mean(d_real)
    return [d_loss, g_loss]

linear_projection_iterator=0
def linear_projection(net, axis=1):
    global linear_projection_iterator
    net = linear(net, 1, scope="d_standard_gan_lin_proj"+str(linear_projection_iterator))
    #net = layer_norm_1(int(net.get_shape()[0]), name='d_standard_gan_lin_proj_bn')(net)
    #net = tf.tanh(net)
    return net

def sigmoid_kl_with_logits(logits, targets):
   # broadcasts the same target value across the whole batch
   # this is implemented so awkwardly because tensorflow lacks an x log x op
   assert isinstance(targets, float)
   if targets in [0., 1.]:
     entropy = 0.
   else:
     entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
     return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy
