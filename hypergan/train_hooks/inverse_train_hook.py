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

    invert: for
        D(-inverse x, inverse x)
    """
    def __init__(self, gan=None, config=None):
        super().__init__(config=config, gan=gan)
        self.gamma = self.config.gamma or [4.0, 4.0]
        if type(self.gamma) == float:
            self.gamma = [self.gamma, self.gamma]
        self.target_x = None
        self.target_g = None
        self.loss = self.gan.create_component("loss")

    def forward(self, d_loss, g_loss):
        if self.target_x == None:
            self.target_g = [Parameter(g, requires_grad=True) for g in self.gan.discriminator_fake_inputs()[0]]
            self.target_x = [Parameter(x, requires_grad=True) for x in self.gan.discriminator_real_inputs()]

        for target, data in zip(self.target_x, self.gan.discriminator_real_inputs()):
            target.data = data.clone()

        if self.config.invert:
            for target, data in zip(self.target_g, self.gan.discriminator_real_inputs()):
                target.data = data.clone()
            neg_inverse_real = self.inverse(self.gan.d_fake, self.gan.forward_discriminator(self.target_g), self.target_g)
            neg_inverse_fake = self.inverse(self.gan.forward_discriminator(self.target_x), self.gan.d_fake, self.target_x)

            reg_fake, g_ = self.loss.forward(self.gan.forward_discriminator(neg_inverse_real), self.gan.forward_discriminator(neg_inverse_fake))
            
            self.add_metric("iig_loss", g_)
            self.add_metric("iid_loss_fake", reg_fake)

            return self.gamma[0]*reg_fake, (self.config.g_gamma or 0.1) * g_

        for target, data in zip(self.target_g, self.gan.discriminator_fake_inputs()[0]):
            target.data = data.clone()

        inverse_fake = self.inverse(self.gan.d_real, self.gan.forward_discriminator(self.target_g), self.target_g)
        inverse_real = self.inverse(self.gan.forward_discriminator(self.target_x), self.gan.d_fake, self.target_x)

        reg_fake, g_ = self.loss.forward(self.gan.forward_discriminator(inverse_fake), self.gan.forward_discriminator(self.gan.discriminator_fake_inputs()[0]))
        reg_real = self.loss.forward(self.gan.forward_discriminator(self.gan.discriminator_real_inputs()), self.gan.forward_discriminator(inverse_real))[0]

        self.add_metric("ig_loss", g_)
        self.add_metric("id_loss_real", reg_fake)
        self.add_metric("id_loss_fake", reg_fake)
        return self.gamma[0]*reg_fake+self.gamma[1]*reg_real, (self.config.g_gamma or 0.1) * g_

    def inverse(self, d_real, d_fake, target):
        loss = self.loss.forward(d_fake, d_real)[0]
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True, only_inputs=True)
        if self.config.inverse_type == 1:
            return [_t/_t.norm() * 10.0 + _d1 for _d1, _t in zip(d1_grads, target)]
        else:
            return [_t + _d1 for _d1, _t in zip(d1_grads, target)]
