#https://arxiv.org/pdf/1811.07296.pdf
import hyperchamber as hc
from functools import reduce

from hypergan.losses.base_loss import BaseLoss

class QPLoss(BaseLoss):
    """https://arxiv.org/abs/1811.07296"""
    def _forward(self, d_real, d_fake):
        gan = self.gan

        pq = d_real
        pp = d_fake

        lam = 10.0/(reduce(lambda x,y:x*y, gan.output_shape()))
        dist = (gan.generator.sample - self.gan.inputs.sample).abs().mean()

        dl = - d_real + d_fake
        d_norm = 10 * dist
        d_loss = ( dl + 0.5 * dl**2 / d_norm).mean()

        g_loss = d_real - d_fake

        return [d_loss, g_loss]
