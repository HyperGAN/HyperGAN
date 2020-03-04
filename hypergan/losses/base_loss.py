from hypergan.gan_component import GANComponent
import numpy as np

class BaseLoss(GANComponent):
    def __init__(self, gan, config):
        super(BaseLoss, self).__init__(gan, config)

    def create(self, *args):
        pass

    def required(self):
        return "".split()

    def forward(self, d_real, d_fake):
        d_loss, g_loss = [c.mean() for c in self._forward(d_real, d_fake)]

        return d_loss, g_loss
