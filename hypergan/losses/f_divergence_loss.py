import tensorflow as tf
import numpy as np
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

TINY=1e-8
class FDivergenceLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config

        gfx = None
        gfg = None

        pi = config.pi or 2
        alpha = config.alpha or 0.5

        if config.type == 'kl':
            gfx = d_real
            gfg = d_fake
        elif config.type == 'js':
            gfx = np.log(2) - tf.log(1+tf.exp(-d_real))
            gfg = np.log(2) - tf.log(1+tf.exp(-d_fake))
        elif config.type == 'js_weighted':
            gfx = -pi*tf.log(pi) - tf.log(1+tf.exp(-d_real))
            gfg = -pi*tf.log(pi) - tf.log(1+tf.exp(-d_fake))
        elif config.type == 'gan':
            log2 = tf.constant(np.log(2)-TINY, dtype=tf.float32)
            d_real = tf.minimum(log2, d_real)
            d_fake = tf.minimum(log2, d_fake)
            gfx = -tf.log(1+tf.exp(-d_real)+TINY)
            gfg = -tf.log(1+tf.exp(-d_fake)+TINY)
        elif config.type == 'reverse_kl':
            gfx = -tf.exp(-d_real)
            gfg = -tf.exp(-d_fake)
        elif config.type == 'pearson' or config.type == 'jeffrey' or config.type == 'alpha2':
            gfx = d_real
            gfg = d_fake
        elif config.type == 'squared_hellinger' or config.type == 'neyman':
            gfx = 1-tf.exp(-d_real)
            gfg = 1-tf.exp(-d_fake)

        elif config.type == 'total_variation':
            gfx = 0.5*tf.nn.tanh(d_real)
            gfg = 0.5*tf.nn.tanh(d_fake)

        elif config.type == 'alpha1':
            gfx = 1./(1-alpha) - tf.log(1+tf.exp(-d_real))
            gfg = 1./(1-alpha) - tf.log(1+tf.exp(-d_fake))

        else:
            raise "Unknown type " + config.type

        conjugate = None

        if config.type == 'kl':
            c = tf.constant(8-TINY, dtype=tf.float32)
            gfg = tf.maximum(gfg, c)
            conjugate = tf.exp(gfg-1)
        elif config.type == 'js':
            c = tf.constant(2-TINY, dtype=tf.float32)
            gfg = tf.maximum(tf.exp(gfg), c)
            conjugate = -tf.log(2-gfg)
        elif config.type == 'js_weighted':
            c = tf.constant(-pi*tf.log(pi)-TINY, dtype=tf.float32)
            gfg = tf.maximum(gfg, c)
            conjugate = (1-pi)*tf.log((1-pi)/((1-pi)*tf.exp(gfg/pi)))
        elif config.type == 'gan':
            gfg = tf.minimum(-TINY, gfg)
            conjugate = -tf.log(1-tf.exp(gfg)+TINY)
        elif config.type == 'reverse_kl':
            gfg = tf.minimum(TINY, gfg)
            conjugate = -1-tf.log(-gfg)
        elif config.type == 'pearson':
            conjugate = 0.25 * tf.square(gfg)+gfg
        elif config.type == 'neyman':
            gfg = tf.minimum(1-TINY, gfg)
            conjugate = 2 - 2 * tf.sqrt(1 - gfg)
        elif config.type == 'squared_hellinger':
            gfg = tf.minimum(1-TINY, gfg)
            conjugate = gfg/(1-gfg)
        elif config.type == 'jeffrey':
            raise "jeffrey conjugate not implemented"

        elif config.type == 'alpha1':
            c = tf.constant(1./(1.-alpha)-TINY, dtype=tf.float32)
            gfg = tf.minimum(gfg, c)
            conjugate = tf.pow(1./alpha * (gfg * ( alpha - 1) + 1), alpha/(alpha - 1.)) - 1. / alpha
        elif config.type == 'alpha2':
            conjugate = tf.pow(1./alpha * (gfg * ( alpha - 1) + 1), alpha/(alpha - 1.)) - 1. / alpha

        elif config.type == 'total_variation':
            gfg = tf.minimum(0.5, tf.maximum(net, -0.5))
            conjugate = gfg
        else:
            raise "Unknown type " + config.type

        gf_threshold  = None # f' in the paper

        if config.type == 'kl':
            gf_threshold = 1
        elif config.type == 'js':
            gf_threshold = 0
        elif config.type == 'gan':
            gf_threshold = -np.log(2)
        elif config.type == 'reverse_kl':
            gf_threshold = -1
        elif config.type == 'pearson':
            gf_threshold = 0
        elif config.type == 'squared_hellinger':
            gf_threshold = 0

        self.gf_threshold=gf_threshold

        d_loss = -gfx+conjugate
        g_loss = -conjugate

        if config.g_loss_type == 'gan':
            g_loss = -conjugate
        elif config.g_loss_type == 'reverse_kl':
            g_loss = -d_fake
        elif config.g_loss_type == 'kl':
            g_loss = d_fake * tf.exp(d_fake)
        elif config.g_loss_type == 'alpha1' or config.g_loss_type == 'alpha2':
            a = alpha
            g_loss = (1.0/(a*(a-1))) * (tf.exp(a*d_fake) - 1 - a*(tf.exp(d_fake) - 1))
        else:
            raise "Unknown type " + config.type

        self.gfg = gfg
        self.gfx = gfx

        return [d_loss, g_loss]

    def g_regularizers(self):
        return []
    def d_regularizers(self):
        return []
