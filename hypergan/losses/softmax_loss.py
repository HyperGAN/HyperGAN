import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

class SoftmaxLoss(BaseLoss):
    """https://arxiv.org/abs/1704.06191"""

    def _forward(self, d_real, d_fake):
        ln_zb = (((-d_real).exp().sum()+(-d_fake).exp().sum())+1e-12).log()

        d_target = 1.0 / d_real.shape[0]
        g_target = d_target / 2.0

        g_loss = g_target * (d_fake.sum() + d_real.sum()) + ln_zb
        d_loss = d_target * d_real.sum() + ln_zb

        return [d_loss, g_loss]

