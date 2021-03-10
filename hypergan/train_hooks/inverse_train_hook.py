import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class InverseTrainHook(BaseTrainHook):
    """

    Adds the terms:

        D(inverse g, g) + D(x, inverse x)
    """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.gamma = self.config.gamma or 4.0
        self.target_x = None
        self.target_g = None
        self.loss = self.gan.create_component("loss")

    def forward(self, d_loss, g_loss):
        if self.target_x == None:
            self.target_x = [Parameter(x, requires_grad=True) for x in self.gan.discriminator_real_inputs()]
            self.target_g = [Parameter(g, requires_grad=True) for g in self.gan.discriminator_fake_inputs()[0]]
        for target, data in zip(self.target_g, self.gan.discriminator_fake_inputs()[0]):
            target.data = data.clone()
        for target, data in zip(self.target_x, self.gan.discriminator_real_inputs()):
            target.data = data.clone()
        inverse_fake = self.inverse(self.gan.forward_discriminator(self.target_g), self.gan.d_real, self.target_g)
        reg_fake = self.loss.forward(self.gan.forward_discriminator(inverse_fake), self.gan.d_fake)[0]
        inverse_real = self.inverse(self.gan.d_fake, self.gan.forward_discriminator(self.target_x), self.target_x)
        reg_real = self.loss.forward(self.gan.d_real, self.gan.forward_discriminator(inverse_real))[0]
        return self.gamma*(reg_fake + reg_real), None

    def inverse(self, d_real, d_fake, target):
        loss = self.loss.forward(d_real, d_fake)[0]
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True)
        return [_d1 + _t for _d1, _t in zip(d1_grads, target)]
