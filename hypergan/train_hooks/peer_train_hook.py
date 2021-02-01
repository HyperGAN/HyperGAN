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
        beta = 0.5
        xj = self.gan.inputs.next()
        xp1 = self.gan.inputs.next()
        xp2 = self.gan.inputs.next()
        di = self.gan.discriminator
        dj = self.discriminator
        alpha = 0.3
        di_x = di(self.gan.augmented_x)
        dj_x = dj(self.gan.augmented_x)
        di_xj = di(xj)
        dj_xj = dj(xj)
        lce1=self.lce(di_x, dj_x)
        lce1b=self.lce(di_xj, dj_xj)
        lce2=alpha * self.lce(di(xp1), dj(xp2))
        lce3=self.lce(dj_x, di_x)
        lce3b=self.lce(dj_xj, di_xj)
        lce4=alpha * self.lce(dj(xp1), di(xp2))
        peer_loss = beta*(lce1-lce2)
        peer_loss += beta*(lce1b-lce2)
        peer_loss /= 2.0
        d_loss2, g_loss = self.loss.forward(self.discriminator(self.gan.augmented_x), self.discriminator(self.gan.augmented_g))
        peer_loss2 = beta*(lce3-lce4)
        peer_loss2 += beta*(lce3b-lce4)
        peer_loss2 /= 2.0

        self.gan.add_metric('g_loss2', g_loss.mean())
        self.gan.add_metric('peer_d', peer_loss.mean())
        self.gan.add_metric('peer_d2', peer_loss2.mean())
        self.gan.add_metric('d_loss2', d_loss2.mean())
        return [peer_loss.mean() + d_loss2 + peer_loss2.mean(), g_loss]

    def discriminator_components(self):
        return [self.discriminator]

    def lce(self, di, dj):
        criterion = torch.nn.BCEWithLogitsLoss()
        mask = torch.greater(self.sigmoid(dj), 0.5).float()
        #return (mask*criterion(di, torch.ones_like(di)) + (1-mask)*criterion(di, torch.zeros_like(di)))
        return mask * torch.log(self.sigmoid(di) + 1e-8) + (1 - mask) * torch.log(1 - self.sigmoid(di) + 1e-8)
