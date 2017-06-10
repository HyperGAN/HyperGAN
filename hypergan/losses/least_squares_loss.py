import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class LeastSquaresLoss(BaseLoss):

    def required(self):
        return "labels".split()

    def _create(self, d_real, d_fake):
        ops = self.gan.ops
        config = self.config

        d_real = config.reduce(d_real, axis = 1)
        d_fake = config.reduce(d_fake, axis = 1)

        a,b,c = config.labels
        square = ops.lookup('square')
        d_loss = 0.5*square(d_real - b) + 0.5*square(d_fake - a)
        g_loss = 0.5*square(d_fake - c)

        return [d_loss, g_loss]
