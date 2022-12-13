import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn as nn
from torch.nn import functional as F

import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class SelfSupervisedTrainHook(BaseTrainHook):
    def __init__(self, gan=None, config=None, trainer=None):
        super().__init__(config=config, gan=gan)

    def forward(self, d_loss, g_loss):
        prediction = self.gan.discriminator.context["rotation"]
        target = self.k.view([self.k.shape[0]])
        loss = F.cross_entropy(prediction, target)# * 0.05
        self.gan.add_metric('ss', loss)
        return [loss, loss]

    def augment_x(self, x):
        rot = x
        k = torch.randint(4, [x.shape[0],1,1,1], device=x.device, dtype=torch.long)
        current = k.clone().detach()
        for i in range(4):
            current = current - torch.ones_like(k)
            mask = (current > 0).float()
            rot = torch.rot90(x, 1, (-2, -1))
            x = x * (1 - mask) + rot * mask
        self.gan.rotated = x
        self.k = k
        return x

    def augment_g(self, g):
        return g
