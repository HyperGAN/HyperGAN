import torch
import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

class AdversarialNormTrainHook(BaseTrainHook):
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
        reg_fake = self.regularize(False, self.gan.discriminator_fake_inputs()[0], self.target_g, self.gan.d_real, self.gan.forward_discriminator(self.target_g))
        reg_real = self.regularize(True, self.gan.discriminator_real_inputs(), self.target_x, self.gan.forward_discriminator(self.target_x), self.gan.d_fake)
        return (reg_fake + reg_real), None

    def regularize(self, invert, inputs, target, d_real, d_fake):
        loss, norm, mod_target = self.regularize_adversarial_norm(d_real, d_fake, target)
        d_real = self.gan.forward_discriminator(mod_target)
        d_fake = self.gan.forward_discriminator(inputs)
        if invert:
            d_l, _g_l = self.loss.forward(d_fake, d_real)
        else:
            d_l, _g_l = self.loss.forward(d_real, d_fake)
        return self.gamma*d_l

    def regularize_adversarial_norm(self, d_real, d_fake, target):
        loss = self.forward_adversarial_norm(d_real, d_fake)
        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True)
        mod_target = [_d1 + _t for _d1, _t in zip(d1_grads, target)]

        return loss, None, mod_target

    def forward_adversarial_norm(self, d_real, d_fake):
        return self.loss.forward(d_fake, d_real)[0]
