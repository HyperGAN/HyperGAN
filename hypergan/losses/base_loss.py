from hypergan.gan_component import GANComponent
import numpy as np
import torch

class BaseLoss(GANComponent):
    def __init__(self, gan, config):
        super(BaseLoss, self).__init__(gan, config)
        self.relu = torch.nn.ReLU()

    def create(self, *args):
        pass

    def required(self):
        return "".split()

    def forward(self, d_real, d_fake):
        d_loss, g_loss = [c.mean() for c in self._forward(d_real, d_fake)]

        return d_loss, g_loss

    def forward_adversarial_norm(self, d_real, d_fake):
        return self.relu(d_real.mean() - d_fake.mean()) ** 2
        #return torch.abs(d_real.mean() - d_fake.mean())
