from hypergan.gan_component import GANComponent
import numpy as np

class BaseLoss():
    def __init__(self, gan, config):
        self.gan = gan
        self.config = config

    def required(self):
        return "reduce".split()

    def forward(self, d_real, d_fake):
        gan = self.gan

        d_loss, g_loss = [c.mean() for c in self._forward(d_real, d_fake)]

        self.gan.add_metric('d_loss', d_loss)
        self.gan.add_metric('g_loss', g_loss)

        self.sample = [d_loss, g_loss]
        self.d_loss = d_loss
        self.g_loss = g_loss

        return self.sample
