import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss

class WassersteinLoss(BaseLoss):

    def _forward(self, d_real, d_fake):
        config = self.config

        if config.kl:
            # https://arxiv.org/abs/1910.09779
            d_fake_norm = torch.mean(d_fake.exp())+1e-8
            d_fake_ratio = (d_fake.exp()+1e-8) / d_fake_norm
            d_fake = d_fake * d_fake_ratio

        d_loss = -d_real + d_fake
        g_loss = -d_fake

        return [d_loss, g_loss]
