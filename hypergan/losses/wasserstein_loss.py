import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class WassersteinLoss(BaseLoss):

    def create(self):
        gan = self.gan
        config = self.config
        ops = self.ops

        net = gan.discriminator.sample

        s = ops.shape(net)
        net = tf.reshape(net, [s[0], -1])

        d_real, d_fake = self.split_batch(net)

        if(config.reverse):
            d_loss = d_real - d_fake
            g_loss = d_fake
        else:
            d_loss = -d_real + d_fake
            g_loss = -d_fake

        if config.gradient_penalty:
            d_loss += gradient_penalty(gan, config.gradient_penalty)

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

