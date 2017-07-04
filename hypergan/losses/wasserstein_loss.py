import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class WassersteinLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        config = self.config

        print("Initializing Wasserstein loss", config.reverse)
        if(config.reverse):
            d_loss = -d_real + d_fake
            g_loss = -d_fake
        else:
            d_loss = d_real - d_fake
            g_loss = d_fake

        return [d_loss, g_loss]
