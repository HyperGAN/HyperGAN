import tensorflow as tf
from hypergan.losses.common import *
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class LeastSquaresLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        ops = self.ops
        ops.describe("LeastSquaresLoss")
        config = self.config

        d_real = config.reduce(d_real, axis = 1)
        d_fake = config.reduce(d_fake, axis = 1)

        a,b,c = config.labels
        square = ops.lookup('square')
        d_loss = square(d_real - b) + square(d_fake - a)
        g_loss = square(d_fake - c)

        return [d_loss, g_loss]
