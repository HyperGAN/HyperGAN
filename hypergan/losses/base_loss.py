from hypergan.gan_component import GANComponent
import numpy as np
import tensorflow as tf

class BaseLoss(GANComponent):
    def __init__(self, gan, config, discriminator=None):
        GANComponent.__init__(self, gan, config)
        self.discriminator = discriminator

    def create(self):
        gan = self.gan

        d_real = 0
        d_fake = 0
        d_loss, g_loss = self._create(d_real, d_fake)

        self.add_metric('d_loss', d_loss)
        if g_loss is not None:
            self.add_metric('g_loss', g_loss)

        self.sample = [d_loss, g_loss]
        self.d_loss = d_loss
        self.g_loss = g_loss

        return self.sample
