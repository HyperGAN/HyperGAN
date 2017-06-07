import tensorflow as tf
import hyperchamber as hc
import numpy as np

from hypergan.losses.standard_loss import StandardLoss
from hypergan.losses.least_squares_loss import LeastSquaresLoss
from hypergan.losses.wasserstein_loss import WassersteinLoss
from hypergan.losses.base_loss import BaseLoss


class LambGanLoss(BaseLoss):

    def required(self):
        return "label_smooth".split()

    def _create(self, d_real, d_fake):
        config = self.config

        alpha = config.alpha
        beta = config.beta
        wgan_loss_d, wgan_loss_g = WassersteinLoss._create(self, d_real, d_fake)
        lsgan_loss_d, lsgan_loss_g = LeastSquaresLoss._create(self, d_real, d_fake)
        standard_loss_d, standard_loss_g = StandardLoss._create(self, d_real, d_fake)

        total = min(alpha + beta,1)

        d_loss = wgan_loss_d*alpha + lsgan_loss_d*beta + (1-total)*standard_loss_d
        g_loss = wgan_loss_g*alpha + lsgan_loss_g*beta + (1-total)*standard_loss_g

        return [d_loss, g_loss]

