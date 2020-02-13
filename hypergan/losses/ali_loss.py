import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss

class AliLoss(BaseLoss):
    def _forward(self, d_real, d_fake):
        criterion = torch.nn.BCEWithLogitsLoss()
        g_loss = criterion(d_fake, torch.ones_like(d_fake)) + criterion(d_real, torch.zeros_like(d_real))
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))

        return [d_loss, g_loss]
