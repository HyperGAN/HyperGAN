import torch
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
        self.gan.source = torch.zeros_like(gan.inputs.next())
        self.source_x = None

    def forward(self, d_loss, g_loss):
        z = self.gan.latent.z
        if self.source_x is None:
            self.source_x = self.gan.x
        zprime = self.g_to_z(self.gan.g)
        reconstruct_x = self.gan.generator(self.g_to_z(self.gan.x)).clone().detach()
        #m = (zprime + z) / 2
        #jsd = zprime*torch.log(zprime/(m+1e-8)) + z * torch.log(z/(m+1e-8))
        loss = ((z - zprime)**2).mean()
        #z_reconstruct_diff = ((self.g_to_z(self.gan.x) - self.g_to_z(reconstruct_x))**2).mean()*1000
        #self.gan.add_metric("IID1z", z_reconstruct_diff)
        #loss += z_reconstruct_diff
        x_reconstruct_diff = ((self.gan.x - reconstruct_x)**2).mean()
        self.gan.add_metric("IID1x", x_reconstruct_diff)
        loss += x_reconstruct_diff
        loss2 = zprime.mean().abs() + (torch.std(zprime) - 1.0) ** 2
        self.gan.add_metric("IID1", loss)
        self.gan.add_metric("IID2", loss2)
        reconstruct_source_x = self.gan.generator(self.g_to_z(self.source_x)).clone().detach()
        self.gan.decoded = reconstruct_source_x
        self.gan.source = self.source_x

        return None, loss+loss2
    def generator_components(self):
        return [self.g_to_z]
