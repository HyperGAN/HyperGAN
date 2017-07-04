import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class CramerLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config
        
        g_loss = d_real - d_fake
        d_loss = -g_loss

        return [d_loss, g_loss]
