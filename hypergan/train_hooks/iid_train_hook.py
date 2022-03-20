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

class IIDTrainHook(BaseTrainHook):
    """
    https://arxiv.org/abs/2106.00563v1
    """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.g_to_z = gan.create_component("G-1", defn=self.config.g_to_z, input=torch.zeros_like(gan.inputs.next()))
        self.gan.decoded = torch.zeros_like(gan.inputs.next())
        self.gan.g_to_z = self.g_to_z
        self.gan.source = torch.zeros_like(gan.inputs.next())
        self.source_x = None
        if self.config.z_discriminator:
            self.z_discriminator = gan.create_component("z_discriminator", defn=self.config.z_discriminator, input=torch.zeros_like(gan.latent.next()))
            self.z_stable_loss = StableGANLoss(gan=self.gan, gammas=self.config.gammas or [1.0, 10.0, 10.0, 0, 10.0])


    def forward(self, d_loss, g_loss):
        z = self.gan.latent.z
        if self.config.align:
            x = self.gan.inputs.next(1)
        else:
            x = self.gan.x

        if self.source_x is None:
            self.source_x = x
            self.gan.source = self.source_x
        types = ["z-g'(g(z))", "x - g(g'(x))", "mean", "std"]
        if self.config.types:
            types = self.config.types
        #zprime = self.g_to_z(self.gan.g.clone().detach())
        zprime = self.g_to_z(self.gan.g)
        zxprime = self.g_to_z(x)
        reconstruct_x = self.gan.generator(self.g_to_z(x)).clone().detach()

        #m = (zprime + z) / 2
        #jsd = zprime*torch.log(zprime/(m+1e-8)) + z * torch.log(z/(m+1e-8))
        loss = torch.zeros_like(z).mean()
        d_l = None
        for t in types:
            if t == "z-g'(g(z))":
                loss += ((z - zprime)**2).mean()
                self.gan.add_metric("IID1d", loss)
            elif t == 'zdisc3':
                z_loss = self.z_stable_loss.stable_loss(self.z_discriminator, [torch.cat([z,z], axis=1)], [torch.cat([z,zprime], axis=1)])
                d_l = z_loss[0]
                g_l = z_loss[1]
                self.gan.add_metric('zd', d_l)
                self.gan.add_metric('zg', g_l)
                loss += g_l
                self.gan.add_metric("recon", ((reconstruct_x-x)**2).sum())
                #self.gan.add_metric('zw', self.z_discriminator.nn_layers[1][1].weight.sum())

            elif t == "g'(x)-g'(g(g'(x)))":
                z_reconstruct_diff = ((self.g_to_z(x) - self.g_to_z(self.gan.generator(self.g_to_z(x).clone().detach())))**2).mean() * 0.5
                self.gan.add_metric("IID1z", z_reconstruct_diff)
                loss += z_reconstruct_diff
            elif t == "x - g(g'(x))":
                x_reconstruct_diff = ((x - reconstruct_x)**2).mean()
                self.gan.add_metric("IID1x", x_reconstruct_diff)
                loss += x_reconstruct_diff
            elif t == 'mean':
                mean = zprime.mean().abs()
                self.gan.add_metric("IIDmean", mean)
                loss += mean
            elif t == 'std':
                std = (torch.std(zprime) - 1.0) ** 2
                self.gan.add_metric("IIDstd", std)
                loss += std
            else:
                raise ValidationException("Unknown type "+t)

        return d_l, loss
    def generator_components(self):
        return [self.g_to_z]

    def discriminator_components(self):
        if self.config.z_discriminator:
            return [self.z_discriminator]
        return []
