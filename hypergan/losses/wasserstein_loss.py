import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class WassersteinLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        config = self.config
        self.ops.describe("WassersteinLoss")

        d_real = config.reduce(d_real, axis = 1)
        d_fake = config.reduce(d_fake, axis = 1)

        if(config.reverse):
            d_loss = d_real - d_fake
            g_loss = d_fake
        else:
            d_loss = -d_real + d_fake
            g_loss = -d_fake

        return [d_loss, g_loss]
