import torch
import torch.nn as nn
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class PeerTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.discriminator = gan.create_component("discriminator")
        self.sigmoid = nn.Sigmoid()
        self.loss = self.gan.initialize_component("loss")

    def forward(self, d_loss, g_loss):
        beta = 1.0
        fraction = 2
        xj = self.gan.inputs.next()
        xp = self.gan.inputs.next()
        di = self.gan.discriminator
        dj = self.discriminator
        alpha = 0.5
        lce1=self.lce(di(xj), dj(xj))
        lce2=alpha * self.lce(di(xp), dj(xp))
        d_loss = beta*fraction*(lce1-lce2)
        d_loss2, g_loss = self.loss.forward(self.discriminator(self.gan.augmented_x), self.discriminator(self.gan.augmented_g))

        self.gan.add_metric('peer_g', g_loss.mean())
        self.gan.add_metric('peer_d', d_loss.mean())
        return [d_loss.mean() + d_loss2, g_loss]

    def discriminator_components(self):
        return [self.discriminator]

    def lce(self, di, dj):
        mask = torch.greater(dj, 0.5).float()
        return mask * torch.log(self.sigmoid(di) + 1e-8) + (1 - mask) * (torch.log(1 - self.sigmoid(di) + 1e-8))
