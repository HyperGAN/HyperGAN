import tensorflow as tf
import hyperchamber as hc
from hypergan.ops.tensorflow.activations import minmaxzero

from hypergan.losses.base_loss import BaseLoss

class BoundaryEquilibriumLoss(BaseLoss):
    def required(self):
        return "type use_k reduce k_lambda gamma initial_k".split()

    # boundary equilibrium gan
    def began(self, d_real, d_fake):
        gan = self.gan
        config = self.config
        ops = self.gan.ops

        a,b,c = config.labels or [0,1,1]

        d_real = config.reduce(d_real)
        d_fake = config.reduce(d_fake)

        k = tf.get_variable(gan.ops.generate_scope()+'k', [1], initializer=tf.constant_initializer(config.initial_k), dtype=config.dtype)

        if config.type == 'wgan':
            l_x = d_real
            l_dg =-d_fake
            g_loss = d_fake
        elif config.type == 'least-squares':
            l_x = tf.square(d_real-b)
            l_dg = tf.square(d_fake - a)
            g_loss = tf.square(d_fake - c)
        else:
            print("No loss defined.  Get ready to crash")

        if config.use_k:
            d_loss = l_x+k*l_dg
        else:
            d_loss = l_x+l_dg

        gamma = config.gamma or 0.5
        gamma_d_real = gamma*d_real

        ### VERIFY FROM HERE
        k_loss = gamma_d_real - g_loss
        clip = k + config.k_lambda * k_loss
        clip = tf.clip_by_value(clip, 0, 1)
        clip = tf.reduce_mean(clip, axis=0)
        update_k = tf.assign(k, tf.reshape(clip, [1]))
        measure = self.gan.ops.squash(l_x + tf.abs(k_loss))

        d_loss = ops.reshape(d_loss, [])
        g_loss = ops.reshape(g_loss, [])

        return [k, update_k, measure, d_loss, g_loss]


    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config

        x = gan.inputs.x
        k, update_k, measure, d_loss, g_loss = self.began(d_real, d_fake)

        self.metrics = {
            'k': k,
            'update_k': update_k, #side effect, this actually trains k
            'measure': measure
        }

        return [d_loss, g_loss]
