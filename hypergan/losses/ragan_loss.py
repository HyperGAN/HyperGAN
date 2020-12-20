import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss
import torch.nn.functional as F

TINY=1e-12
class RaganLoss(BaseLoss):
    """https://arxiv.org/abs/1807.00734"""
    def __init__(self, gan, config):
        super(RaganLoss, self).__init__(gan, config)
        self.sigmoid = torch.nn.Sigmoid()

    def required(self):
        return "".split()

    def _forward(self, d_real, d_fake):
        config = self.config
        gan = self.gan
        loss_type = self.config.type or "standard"

        if config.rgan:
            cr = d_real
            cf = d_real
        else:
            cr = torch.mean(d_real,0)
            cf = torch.mean(d_fake,0)

        if loss_type == "least_squares":
            a,b,c = (config.labels or [-1,1,1])
            d_loss = 0.5*(d_real - cf - b)**2 + 0.5*(d_fake - cr - a)**2
            g_loss = 0.5*(d_fake - cr - c)**2 + 0.5*(d_real - cf - a)**2
        elif loss_type == "hinge":
            d_loss = torch.clamp(1-(d_real - cf), min=0)+torch.clamp(1+(d_fake-cr), min=0)
            g_loss = torch.clamp(1-(d_fake - cr), min=0)+torch.clamp(1+(d_real-cf), min=0)
        elif loss_type == "wasserstein":
            d_loss = -(d_real-cf) + (d_fake-cr)
            g_loss = -(d_fake-cr)
        elif loss_type == "standard":
            criterion = torch.nn.BCEWithLogitsLoss()
            g_loss = criterion(d_fake-cr, torch.ones_like(d_fake))
            d_loss = criterion(d_real-cf, torch.ones_like(d_real)) + criterion(d_fake-cr, torch.zeros_like(d_fake))

        return [d_loss, g_loss]
