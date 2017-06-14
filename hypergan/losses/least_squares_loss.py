import tensorflow as tf
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class LeastSquaresLoss(BaseLoss):

    def required(self):
        return "labels".split()

    def _create(self, d_real, d_fake):
        ops = self.gan.ops
        config = self.config

        a,b,c = config.labels
        square = ops.lookup('square')
        d_loss = 0.5*square(d_real - b) + 0.5*square(d_fake - a)
        g_loss = 0.5*square(d_fake - c)

        return [d_loss, g_loss]
