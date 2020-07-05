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
    def __init__(self, gan=None, config=None, trainer=None):
        super().__init__(config=config, gan=gan, trainer=trainer)
        self.d_loss = None
        self.g_loss = None
        if self.config.gamma is not None:
            self.gamma = torch.Tensor([self.config.gamma]).float()[0].cuda()#self.gan.configurable_param(self.config.gamma or 1.0)
        if self.config.gammas is not None:
            self.gammas = [
                        torch.Tensor([self.config.gammas[0]]).float()[0].cuda(),#self.gan.configurable_param(self.config.gamma or 1.0)
                        torch.Tensor([self.config.gammas[1]]).float()[0].cuda()#self.gan.configurable_param(self.config.gamma or 1.0)
                    ]
        self.relu = torch.nn.ReLU()
        self.target = [Parameter(x, requires_grad=True) for x in self.gan.discriminator_real_inputs()]
        self.x_mod_target = torch.zeros_like(self.target[0])
        self.g_mod_target = torch.zeros_like(self.target[0])

    def forward(self, d_loss, g_loss):
        if self.config.mode == "real" or self.config.mode is None:
            for target, data in zip(self.target, self.gan.discriminator_real_inputs()):
                target.data = data.clone()
            d_fake = self.gan.d_fake
            d_real = self.gan.forward_discriminator(self.target)
            loss, _, mod_target = self.regularize_adversarial_norm(d_fake, d_real, self.target)
            norm = self.relu(self.config.offset-((mod_target[0] - self.gan.x) ** 2))
        elif self.config.mode == "fake":
            for target, data in zip(self.target, self.gan.discriminator_fake_inputs()):
                target.data = data.clone()
            d_fake = self.gan.forward_discriminator(self.target)
            d_real = self.gan.d_real
            loss, norm, mod_target = self.regularize_adversarial_norm(d_real, d_fake, self.target)
            norm = self.relu(self.config.offset-((mod_target[0] - self.gan.g) ** 2))

        if loss is None:
            return [None, None]

        if self.config.loss:
          if "g" in self.config.loss:
              self.g_loss = self.gamma * norm.mean()
              self.gan.add_metric('an_g', self.g_loss)
          if "d" in self.config.loss:
              self.d_loss = self.gamma * norm.mean()
              self.gan.add_metric('an_d', self.d_loss)
          if "dg" in self.config.loss:
              self.d_loss = self.gammas[0] * norm.mean()
              self.gan.add_metric('an_d', self.d_loss)
              self.g_loss = self.gammas[1] * norm.mean()
              self.gan.add_metric('an_g', self.g_loss)
        else:
            self.d_loss = self.gamma * norm.mean()
            self.gan.add_metric('an_d', self.d_loss)

        return [self.d_loss, self.g_loss]

    def regularize_adversarial_norm(self, d1_logits, d2_logits, target):
        loss = self.gan.loss.forward_adversarial_norm(d1_logits, d2_logits)

        if loss == 0:
            return [None, None, None]

        d1_grads = torch_grad(outputs=loss, inputs=target, retain_graph=True, create_graph=True)
        d1_norm = [torch.norm(_d1_grads.view(-1),p=2,dim=0) for _d1_grads in d1_grads]
        reg_d1 = d1_norm[0]
        for d1 in d1_norm[1:]:
            reg_d1 = reg_d1 + d1
        mod_target = [_d1 + _t for _d1, _t in zip(d1_grads, target)]

        return loss, reg_d1, mod_target
