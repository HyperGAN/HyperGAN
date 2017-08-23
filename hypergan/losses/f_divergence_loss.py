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

        if config.type == 'kl':
            gfx = d_real
            gfg = d_fake
        elif config.type == 'js':
            gfx = np.log(2) - tf.log(1+tf.exp(-d_real))
            gfg = np.log(2) - tf.log(1+tf.exp(-d_fake))
        elif config.type == 'gan':
            gfx = -tf.log(1+tf.exp(-d_real)+TINY)
            gfg = -tf.log(1+tf.exp(-d_fake)+TINY)
        elif config.type == 'reverse_kl':
            gfx = -tf.exp(-d_real)
            gfg = -tf.exp(-d_fake)
        elif config.type == 'pearson':
            gfx = d_real
            gfg = d_fake
        elif config.type == 'squared_hellinger':
            gfx = 1-tf.exp(-d_real)
            gfg = 1-tf.exp(-d_fake)

        
        conjugate = None

        if config.type == 'kl':
            conjugate = tf.exp(gfg-1)
        elif config.type == 'js':
            conjugate = -tf.log(2-tf.exp(gfg))
        elif config.type == 'gan':
            conjugate = -tf.log(1-tf.exp(gfg)+TINY)
        elif config.type == 'reverse_kl':
            conjugate = -1-tf.log(-gfg)
        elif config.type == 'pearson':
            conjugate = 0.25 * tf.square(gfg)+gfg
        elif config.type == 'squared_hellinger':
            conjugate = gfg/(1-gfg+TINY)


        gfg_threshold  = None # f' in the paper

        if config.type == 'kl':
            gfg_threshold = 1
        elif config.type == 'js':
            gfg_threshold = 0
        elif config.type == 'gan':
            gfg_threshold = -np.log(2)
        elif config.type == 'reverse_kl':
            gfg_threshold = -1
        elif config.type == 'pearson':
            gfg_threshold = 0
        elif config.type == 'squared_hellinger':
            gfg_threshold = 0

        d_loss = -gfx+conjugate
        g_loss = -conjugate

        return [d_loss, g_loss]
