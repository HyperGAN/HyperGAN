#From https://gist.github.com/EndingCredits/b5f35e84df10d46cfa716178d9c862a3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import hyperchamber as hc
import numpy as np
import inspect
from operator import itemgetter
from hypergan.train_hooks.base_train_hook import BaseTrainHook

class StabilizingTrainingTrainHook(BaseTrainHook):
  """ https://github.com/rothk/Stabilizing_GANs """
  def __init__(self, gan=None, config=None, trainer=None):
      super().__init__(config=config, gan=gan, trainer=trainer)
      self.d_loss = None
      self.g_loss = None
      self.sig = torch.nn.Sigmoid().cuda()
      self.gamma = self.gan.configurable_param(self.config.gamma or 1.0)

  def forward(self):
      x = self.gan.inputs.sample
      g = self.gan.generator_sample
      d1_params = Variable(x, requires_grad=True).cuda()#self.gan.d_parameters()
      d2_params = Variable(g, requires_grad=True).cuda()#self.gan.g_parameters()
      d1_logits = self.gan.discriminator(d1_params)
      d2_logits = self.gan.discriminator(d2_params)
      d1 = self.sig(d1_logits)
      d2 = self.sig(d2_logits)
      d1_params = list(self.gan.discriminator.parameters()) + [d1_logits]
      d2_params = list(self.gan.discriminator.parameters()) + [d2_logits]
      d1_grads = torch_grad(outputs=d1_logits.mean(), inputs=d1_params, retain_graph=True, create_graph=True)
      d2_grads = torch_grad(outputs=d2_logits.mean(), inputs=d2_params, retain_graph=True, create_graph=True)
      d1_norm = [torch.norm(_d1_grads.view(-1).cuda(),p=2,dim=0) for _d1_grads in d1_grads]
      d2_norm = [torch.norm(_d2_grads.view(-1).cuda(),p=2,dim=0) for _d2_grads in d2_grads]

      reg_d1 = [(((1.0-d1)**2).cuda() * (_d1_norm**2).cuda()) for _d1_norm in d1_norm]
      reg_d2 = [((d2**2).cuda() * (_d2_norm**2).cuda()) for _d2_norm in d2_norm]
      #reg_d1 = [((d1**2).cuda() * (_d1_norm**2).cuda()) for _d1_norm in d1_norm]
      #reg_d2 = [(((1.0-d2)**2).cuda() * (_d2_norm**2).cuda()) for _d2_norm in d2_norm]
      reg_d1 = sum(reg_d1)
      reg_d2 = sum(reg_d2)
      #self.d_loss = self.gamma * reg_d1.mean()
      self.d_loss = self.gamma * (reg_d1+reg_d2).mean()
      #self.g_loss = self.gamma * reg_d2.mean()
      #self.gan.add_metric('stable_js', self.ops.squash(self.d_loss))

      return [self.d_loss, self.g_loss]
