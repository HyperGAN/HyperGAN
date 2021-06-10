import hyperchamber as hc
import torch

from hypergan.losses.base_loss import BaseLoss
import torch.nn.functional as F

class StandardLoss(BaseLoss):
    def __init__(self, gan, config):
        super(StandardLoss, self).__init__(gan, config)
        #self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.two = torch.Tensor([2.0]).cuda()
        self.eps = torch.Tensor([1e-12]).cuda()

    def _forward(self, d_real, d_fake):
        criterion = torch.nn.BCEWithLogitsLoss()
        #g_loss = criterion(d_fake, torch.ones_like(d_fake))
        #d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(d_fake, torch.zeros_like(d_fake))
        g_loss = torch.log(1-self.sigmoid(d_fake) + 1e-13)
        d_loss = torch.log(self.sigmoid(d_real) + 1e-13) +torch.log(1-self.sigmoid(d_fake) + 1e-13)
        d_loss = -d_loss

        return [d_loss, g_loss]

