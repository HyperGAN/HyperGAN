import tensorflow as tf
import hyperchamber as hc
from hypergan.generators.common import minmaxzero
from hypergan.losses.common import *

from hypergan.losses.base_loss import BaseLoss

class BoundaryEquilibriumLoss(BaseLoss):
    def required(self):
        return "type use_k reduce k_lambda gamma initial_k labels".split()

    def g(gan, z):
        #reuse variables
        with(tf.variable_scope("generator", reuse=True)):
            # call generator
            generator = hc.Config(hc.lookup_functions(gan.config.generator))
            nets = generator.create(generator, gan, z)
            return nets[0]

    def loss(gan, x, reuse=True):
        for i, discriminator in enumerate(gan.config.discriminators):
            discriminator = hc.Config(hc.lookup_functions(discriminator))
            with(tf.variable_scope("discriminator", reuse=reuse)):
                ds = discriminator.create(gan, discriminator, x, x, gan.graph.xs, gan.graph.gs,prefix="d_"+str(i))
                bs = gan.config.batch_size
                net = ds
                net = tf.slice(net, [0,0],[bs, -1])
                print('net is', net)
                return tf.reduce_mean(net, axis=1)


    # boundary equilibrium gan
    def began(self, gan, config, d_real, d_fake, prefix=''):
        d_fake = config.reduce(d_fake, axis=1)
        d_real = config.reduce(d_real, axis=1)
        a,b,c = config.labels

        k = tf.get_variable(prefix+'k', [1], initializer=tf.constant_initializer(config.initial_k), dtype=config.dtype)

        if config.type == 'wgan':
            l_x = d_real
            l_dg =-d_fake
            g_loss = d_fake
        else:
            l_x = tf.square(d_real-b)
            l_dg = tf.square(d_fake - a)
            g_loss = tf.square(d_fake - c)

        if config.use_k:
            d_loss = l_x+k*l_dg
        else:
            d_loss = l_x+l_dg

        if config.gradient_penalty:
            d_loss += gradient_penalty(gan, config.gradient_penalty)

        gamma = config.gamma * tf.ones_like(d_fake)

        if config.use_k:
            gamma_d_real = gamma*d_real
        else:
            gamma_d_real = d_real
        k_loss = tf.reduce_mean(gamma_d_real - d_fake, axis=0)
        update_k = tf.assign(k, minmaxzero(k + config.k_lambda * k_loss))
        measure = tf.reduce_mean(l_x + tf.abs(k_loss), axis=0)
     
        return [k, update_k, measure, d_loss, g_loss]


    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config

        x = gan.graph.x
        k, update_k, measure, d_loss, g_loss = self.began(gan, config, d_real, d_fake)
        gan.graph.measure = measure
        gan.graph.k = k
        gan.graph.update_k = update_k

        gan.graph.gamma = config.gamma

        self.metrics = {
            'd_loss': d_loss,
            'g_loss': g_loss,
            'k': k
        }


        return [d_loss, g_loss]
