import torch
from hypergan.losses.stable_gan_loss import StableGANLoss
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class AutoencodeTrainHook(BaseTrainHook):
    """
    """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.g_to_z = gan.create_component("G-1", defn=self.config.g_to_z, input=torch.zeros_like(gan.inputs.next()))
        self.gan.decoded = torch.zeros_like(gan.inputs.next())
        self.gan.g_to_z = self.g_to_z
        self.gan.source = torch.zeros_like(gan.inputs.next())
        self.source_x = None
        self.ae_discriminator = gan.create_component("ae_discriminator", defn=self.config.ae_discriminator, input=torch.zeros_like(gan.latent.next()))
        self.ae_stable_loss = StableGANLoss(gan=self.gan, gammas=self.config.gammas, offsets=self.config.offsets, metric_name="z")


    def forward(self, d_loss, g_loss):
        z = self.gan.latent.z
        if self.config.align:
            x = self.gan.inputs.next(1)
        else:
            x = self.gan.x

        if self.source_x is None:
            self.source_x = x
            self.gan.source = self.source_x
        if self.config.types:
            types = self.config.types
        zprime = self.g_to_z(self.gan.g)
        zxprime = self.g_to_z(x)
        reconstruct_x = self.gan.generator(self.g_to_z(x))
        g = self.gan.g
        reconstruct_g = self.gan.generator(self.g_to_z(g))


        d_l = None
        if self.config.type == 'g':
            z_loss = self.ae_stable_loss.ae_stable_loss(self.ae_discriminator, g, reconstruct_g)
        else:
            z_loss = self.ae_stable_loss.ae_stable_loss(self.ae_discriminator, x, reconstruct_x)
        d_l = z_loss[0]
        g_l = z_loss[1]
        self.gan.add_metric('aed', d_l)
        self.gan.add_metric('aeg', g_l)
        self.gan.add_metric("recon", ((reconstruct_x.clone().detach()-x)**2).sum())

        return d_l, g_l

    def generator_components(self):
        return [self.g_to_z]

    def discriminator_components(self):
        return [self.ae_discriminator]
