import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss

class LogisticLoss(BaseLoss):
    """ported from stylegan"""
    def __init__(self, gan, config):
        super(LogisticLoss, self).__init__(gan, config)
        self.softplus = torch.nn.Softplus(self.config.beta or 1, self.config.threshold or 20)

    def _forward(self, d_real, d_fake):
        d_loss = self.softplus(-d_real) + self.softplus(d_fake)
        g_loss = self.softplus(-d_fake)

        return [d_loss, g_loss]
