import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss
import torch.nn.functional as F

class StandardLoss(BaseLoss):
    def _forward(self, d_real, d_fake):
        criterion = torch.nn.BCEWithLogitsLoss()
        g_loss = criterion(d_fake, torch.ones_like(d_fake))
        d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))

        return [d_loss, g_loss]

    def div(self, p, q):
        return F.sigmoid(p)*torch.log(F.sigmoid(p)/(F.sigmoid(q)+self.eps))
        #return F.sigmoid(p)*torch.log(self.two*F.sigmoid(p)/(F.sigmoid(p)+F.sigmoid(q)+self.eps))
    def forward_adversarial_norm(self, d_real, d_fake):
        return self.div(d_real, d_fake).mean()
